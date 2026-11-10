import ctypes
import functools
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)
_HIP_HOST_MALLOC_NUMA_USER = 0x20000000
_MPOL_INTERLEAVE = 3
_OWNERS = []


@functools.lru_cache(maxsize=1)
def _hcu_runtime():
    import torch
    from hcu_sglang.layers.engram_host import _runtime

    torch.cuda.init()
    hip = _runtime(os.getpid())
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].startswith('/'):
            path = Path(parts[5])
            if path.name.startswith('libhsa-runtime64.so.'):
                paths.add(str(path.resolve()))
    if len(paths) != 1:
        raise RuntimeError('Engram HSA storage requires one loaded HSA runtime')
    hsa = ctypes.CDLL(paths.pop(), mode=os.RTLD_NOLOAD)
    ptr = ctypes.c_void_p
    out_ptr = ctypes.POINTER(ptr)
    handle_ptr = ctypes.POINTER(_HsaIpcHandle)
    signatures = [
        (hip.hipHostMalloc, [out_ptr, ctypes.c_size_t, ctypes.c_uint], ctypes.c_int),
        (hip.hipHostFree, [ptr], ctypes.c_int),
        (hip.hipHostGetDevicePointer, [out_ptr, ptr, ctypes.c_uint], ctypes.c_int),
        (hip.hipGetErrorName, [ctypes.c_int], ctypes.c_char_p),
        (hip.hipGetErrorString, [ctypes.c_int], ctypes.c_char_p),
        (hsa.hsa_amd_ipc_memory_create, [ptr, ctypes.c_size_t, handle_ptr], ctypes.c_int),
        (hsa.hsa_amd_ipc_memory_attach, [handle_ptr, ctypes.c_size_t, ctypes.c_uint, ptr, out_ptr], ctypes.c_int),
        (hsa.hsa_amd_ipc_memory_detach, [ptr], ctypes.c_int),
        (hsa.hsa_status_string, [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)], ctypes.c_int),
    ]
    for fn, args, result in signatures:
        fn.argtypes, fn.restype = args, result
    return hip, hsa


class _HsaIpcHandle(ctypes.Structure):
    # HCU's DTK exposes the hsa_amd_* ABI too; these are runtime symbol names.
    # hsa_amd_ipc_memory_t is a process-independent 256-bit allocation handle.
    _fields_ = [("words", ctypes.c_uint32 * 8)]

def _check_hip(lib, code, operation):
    if code:
        name, detail = lib.hipGetErrorName(code), lib.hipGetErrorString(code)
        raise RuntimeError(
            f"{operation}: HIP code={code}, "
            f"{name.decode() if name else 'unknown'}: "
            f"{detail.decode() if detail else 'unknown'}"
        )

def _check_hsa(lib, code, operation):
    if code:
        detail = ctypes.c_char_p()
        lib.hsa_status_string(code, ctypes.byref(detail))
        raise RuntimeError(
            f"{operation}: HSA code={code}, "
            f"{detail.value.decode() if detail.value else 'unknown'}"
        )

class _NumaPolicy:
    """Save/restore this thread's policy; never change system-wide NUMA settings."""

    def __init__(self):
        self.lib = ctypes.CDLL('libnuma.so.1', use_errno=True)
        if self.lib is None:
            raise RuntimeError("libnuma is required for HCU shared host allocations")
        self.lib.numa_max_possible_node.restype = ctypes.c_int
        self.count = self.lib.numa_max_possible_node() + 1
        if self.count <= 0:
            raise RuntimeError("Cannot determine NUMA node mask size")
        self.bits = ctypes.sizeof(ctypes.c_ulong) * 8
        self.mask_type = ctypes.c_ulong * ((self.count + self.bits - 1) // self.bits)
        self.mode, self.mask = ctypes.c_int(), self.mask_type()
        self.lib.get_mempolicy.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        self.lib.set_mempolicy.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
        ]
        self._check(
            self.lib.get_mempolicy(
                ctypes.byref(self.mode), self.mask, self.count, None, 0
            ),
            "get_mempolicy",
        )
        self.nodes = [
            node
            for node in range(self.count)
            if self.mask[node // self.bits] & (1 << (node % self.bits))
        ]
        if not self.nodes:
            # MPOL_DEFAULT/LOCAL has no explicit mask. Respect the cpuset.
            with open("/proc/self/status") as f:
                allowed = next(
                    line.split(":", 1)[1].strip()
                    for line in f
                    if line.startswith("Mems_allowed_list:")
                )
            for item in allowed.split(","):
                bounds = item.split("-")
                self.nodes.extend(range(int(bounds[0]), int(bounds[-1]) + 1))

    @staticmethod
    def _check(rc, operation):
        if rc:
            err = ctypes.get_errno()
            raise OSError(err, f"{operation}: {os.strerror(err)}")

    @contextmanager
    def interleave(self, nodes):
        mask = self.mask_type()
        for node in nodes:
            mask[node // self.bits] |= 1 << (node % self.bits)
        self._check(
            self.lib.set_mempolicy(_MPOL_INTERLEAVE, mask, self.count),
            "set NUMA interleave",
        )
        try:
            yield
        finally:
            self._check(
                self.lib.set_mempolicy(self.mode.value, self.mask, self.count),
                "restore NUMA policy",
            )

class _HcuSharedAllocation:
    """One pinned host allocation, imported into every local group process.

    HSA_USERPTR_FOR_PAGED_MEM=0 must be set before runtime initialization:
    ordinary USERPTR allocations cannot be exported through HSA IPC. Importing
    one allocation avoids charging the full table against KFD's system-memory
    budget once per rank. Kernels still read the complete table directly.

    Retain this object for the entire lifetime of all tensor views and GPU
    reads. close() is collective and is only safe after all reads have finished.
    """

    def __init__(self, nbytes, group, rank):
        self.group, self.nbytes = group, nbytes
        self.rank = rank
        self.native_status = []
        self.closed = False
        page_size = os.sysconf("SC_PAGE_SIZE")
        self.span = ((nbytes + page_size - 1) // page_size) * page_size
        self.base = ctypes.c_void_p()
        self.hip = self.hsa = None
        self.bytes = None
        self.handle = _HsaIpcHandle()
        _OWNERS.append(self)
        try:
            nodes = self._stage("runtime/NUMA preparation", self._prepare)
            self.nodes = sorted({node for peer_nodes in nodes for node in peer_nodes})
            handles = self._stage("allocation/export", self._allocate)
            self.handle = _HsaIpcHandle.from_buffer_copy(handles[0])
            self._stage("IPC attach/tensor view", self._attach)
        except Exception:
            try:
                self.close()
            except Exception:
                logger.exception("Failed to clean up an HCU shared host allocation")
            raise
        logger.info(
            "Shared host allocation: backend=hsa_ipc, "
            "bytes=%d, numa_nodes=%s, rank=%d/%d",
            nbytes,
            self.nodes,
            self.rank,
            len(nodes),
        )

    def _stage(self, name, action):
        error = value = None
        try:
            value = action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        results = self.group.all_gather_object((error, value))
        failures = [
            f"rank {rank}: {err}" for rank, (err, _) in enumerate(results) if err
        ]
        if failures:
            raise RuntimeError(
                f"HCU shared host allocation {name} ({self.nbytes} bytes): "
                + "; ".join(failures)
            )
        return [value for _, value in results]

    def _prepare(self):
        if os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") != "0":
            raise RuntimeError(
                "HCU shared host allocations require HSA_USERPTR_FOR_PAGED_MEM=0 "
                "in the launch environment before HIP/HSA initialization"
            )
        self.hip, self.hsa = _hcu_runtime()
        self.policy = _NumaPolicy()
        if self.nbytes <= 0:
            raise ValueError("Shared host allocation size must be positive")
        return self.policy.nodes

    def _allocate(self):
        if self.rank != 0:
            return
        # hipHostMallocNumaUser honors the temporary policy. Shared
        # allocations may exceed one NUMA node, so use the group's memory nodes.
        with self.policy.interleave(self.nodes):
            self._call('hip', 'hipHostMalloc', ctypes.byref(self.base), self.span, _HIP_HOST_MALLOC_NUMA_USER)
        self._call('hsa', 'hsa_amd_ipc_memory_create', self.base, self.span, ctypes.byref(self.handle))
        device = ctypes.c_void_p()
        self._call('hip', 'hipHostGetDevicePointer', ctypes.byref(device), self.base, 0)
        self.device_ptr = device.value
        return bytes(self.handle)

    def _attach(self):
        if self.rank != 0:
            self._call('hsa', 'hsa_amd_ipc_memory_attach', ctypes.byref(self.handle), self.span, 0, None, ctypes.byref(self.base))
            self.device_ptr = self.base.value
        self._make_view()

    def _make_view(self):
        import torch
        # frombuffer retains the ctypes view; this allocation owns the storage.
        buffer = (ctypes.c_ubyte * self.nbytes).from_address(self.base.value)
        self.bytes = torch.frombuffer(buffer, dtype=torch.uint8)

    def close(self):
        self.bytes = None

        def detach():
            if self.rank != 0 and self.base.value:
                self._call('hsa', 'hsa_amd_ipc_memory_detach', self.base)
                self.base = ctypes.c_void_p()

        def free():
            if self.rank == 0 and self.base.value:
                self._call('hip', 'hipHostFree', self.base)
                self.base = ctypes.c_void_p()

        # All imports detach before the allocation owner frees its backing.
        self._stage("IPC detach", detach)
        self._stage("allocation release", free)
        self.closed = True

    def _call(self, kind, operation, *args):
        lib = self.hip if kind == 'hip' else self.hsa
        code = getattr(lib, operation)(*args)
        row = {'api': operation, 'type_module': type(code).__module__,
               'type_name': type(code).__qualname__, 'accepted_type': type(code) is int,
               'code': code if type(code) is int else None}
        self.native_status.append(row)
        if type(code) is not int:
            raise TypeError('Native host allocation returned an unknown status type')
        (_check_hip if kind == 'hip' else _check_hsa)(lib, code, operation)
        return code


def around_host_table_init(original_fn, self, layout, nbytes, name, group):
    if not is_hcu() or layout != 'shared':
        return original_fn(self, layout, nbytes, name, group)
    if type(nbytes) is not int or nbytes <= 0:
        raise ValueError('Engram shared allocation size must be a positive integer')
    if os.environ.get('HSA_USERPTR_FOR_PAGED_MEM') != '0':
        raise RuntimeError('Engram HSA shared storage requires HSA_USERPTR_FOR_PAGED_MEM=0 before runtime initialization')
    if not 0 <= group.rank_in_group < group.world_size:
        raise ValueError('Engram HSA shared storage has invalid group placement')
    self.layout, self.nbytes, self.group, self.dirty = layout, nbytes, group, False
    self.fd = self.mm = None
    allocation = _HcuSharedAllocation(nbytes, group, group.rank_in_group)
    self._hcu_hsa_owner = allocation
    self.bytes, self.device_ptr = allocation.bytes, allocation.device_ptr


def around_host_table_finish(original_fn, self, label):
    if not hasattr(self, '_hcu_hsa_owner'):
        return original_fn(self, label)
    if not self.dirty:
        return None
    self.dirty = False
    self.group.barrier()
    logger.info('engram host table %s: layout=shared, backend=hsa_ipc, %.0f MiB shared by %d ranks',
                label, self.nbytes / 2**20, self.group.world_size)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register('sglang.srt.layers.engram._HostTable.__init__', around_host_table_init, HookType.AROUND)
    HookRegistry.register('sglang.srt.layers.engram._HostTable.finish_load', around_host_table_finish, HookType.AROUND)

import ctypes
import os
from functools import lru_cache
from pathlib import Path

from hcu_sglang.utils.detection import is_hcu


def _loaded_runtime_path(maps):
    paths = set()
    for line in maps.splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[5].startswith('/'):
            continue
        path = Path(parts[5])
        if path.name == 'libamdhip64.so' or path.name.startswith(('libamdhip64.so.', 'libgalaxyhip.so.')):
            paths.add(str(path.resolve()))
    if len(paths) != 1:
        raise RuntimeError('Engram host mapping requires one loaded HIP runtime')
    return paths.pop()


@lru_cache(maxsize=1)
def _runtime(pid):
    if pid != os.getpid():
        raise RuntimeError('Engram host mapping runtime belongs to another process')
    path = _loaded_runtime_path(Path('/proc/self/maps').read_text())
    lib = ctypes.CDLL(path)
    lib.hipHostGetDevicePointer.restype = ctypes.c_int
    lib.hipHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    return lib


def around_registered_device_ptr(original_fn, host_ptr):
    if not is_hcu():
        return original_fn(host_ptr)
    if type(host_ptr) is not int or host_ptr <= 0:
        raise ValueError('Engram host pointer must be a positive integer')
    device = ctypes.c_void_p()
    code = _runtime(os.getpid()).hipHostGetDevicePointer(ctypes.byref(device), host_ptr, 0)
    if code != 0 or not device.value:
        raise RuntimeError(f'hipHostGetDevicePointer failed: {code}')
    return device.value


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register('sglang.srt.layers.engram._registered_device_ptr', around_registered_device_ptr, HookType.AROUND)

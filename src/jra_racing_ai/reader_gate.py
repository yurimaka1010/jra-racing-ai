"""One local reader connection at a time, without changing PostgreSQL role limits."""
from contextlib import contextmanager
import hashlib
import os
from .core import SafeAuditError


@contextmanager
def reader_gate(config, timeout_ms=20000):
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000:
        raise SafeAuditError('invalid_configuration')
    if os.name != 'nt':
        yield
        return
    # Named Windows mutexes are released by the OS if the owning process dies;
    # no stale filesystem lock or database write is needed. The scheduled task
    # and viewer use the same existing interactive Windows user/session.
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    api.CreateMutexW.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    api.ReleaseMutex.restype = wintypes.BOOL
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    api.CloseHandle.restype = wintypes.BOOL
    key = hashlib.sha256(f'loopback:{config.port}/{config.database}/{config.user}'.encode()).hexdigest()[:24]
    handle = api.CreateMutexW(None, False, 'Local\\JraRacingAI_RO_' + key)
    if not handle: raise SafeAuditError('connection_failed')
    acquired = False
    try:
        result = api.WaitForSingleObject(handle, timeout_ms)
        if result == 258: raise SafeAuditError('query_timeout')
        if result not in (0, 128): raise SafeAuditError('connection_failed')
        acquired = True
        yield
    finally:
        if acquired: api.ReleaseMutex(handle)
        api.CloseHandle(handle)

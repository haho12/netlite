# Wrapper to make numba optional, as DLL import errors of the LLVM compiler have been observed on windows.
# When numba is not avaliable, the convolutional layer is slowed down significantly.
import warnings

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Numba import error (not installed?): {e}")
    NUMBA_AVAILABLE = False
except OSError as e:
    warnings.warn(f"Numba import error (missing DLL?): {e}")
    NUMBA_AVAILABLE = False

if not NUMBA_AVAILABLE:
    # create fake numba API with no-op decorators
    class _FakeNumba:
        def njit(self, *args, **kwargs):
            def decorator(func):
                return func    # return the original function unchanged
            return decorator

        def prange(self, x):
            return range(x)    # prange → normal range

    numba = _FakeNumba()

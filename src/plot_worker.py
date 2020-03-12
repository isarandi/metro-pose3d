"""Matplotlib messes with the process it runs in. You can't just call it and
generate a plot into an image for example. It would load things
that conflict with other libraries, like OpenCV, Qt etc.
This module comes to the rescue. If you decorate a function with `via_worker_process`, it will
run in a separate worker process. The current thread blocks until the result is back.
"""

import functools
import multiprocessing as mp

import util
from init import FLAGS

_pool = None
_original_functions = {}


def _call_original(wrapped, *args, **kwargs):
    # This runs in the new process
    try:
        _original_functions[wrapped](*args, **kwargs)
    except KeyError:
        raise RuntimeError(f'Problem in the `via_worker_process` decorator mechanism.'
                           f' The worker process doesn\'t know about the '
                           f'original function corresponding to {wrapped}. '
                           f'This is probably a bug.')


def via_worker_process(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        caller = functools.partial(_call_original, wrapped)
        try:
            return _get_pool().apply(caller, args, kwargs)
        except AttributeError as ex:
            if "Can't pickle local object" in str(ex):
                raise RuntimeError(
                    f'`via_worker_process` can only be applied as a decorator to global functions, '
                    f'but was used on {f}')
            raise

    # this line is only relevant when this runs in the worker process
    # since it's the one that will look up `wrapped` in `_original_functions`
    # when executing `_call_original`
    _original_functions[wrapped] = f
    return wrapped


# def _init_worker():
#     # Terminate on parent death:
#     prctl = ctypes.CDLL("libc.so.6").prctl
#     PR_SET_PDEATHSIG = 1
#     prctl(PR_SET_PDEATHSIG, signal.SIGTERM)



def initialize(original_functions, flags):
    # import plotting modules
    # noinspection PyUnresolvedReferences
    import mpl_toolkits.mplot3d.axes3d

    global _original_functions
    import matplotlib.pyplot as plt
    _original_functions = original_functions
    util.init_worker_process_flags(flags)
    plt.switch_backend('TkAgg')


def _get_pool():
    global _pool
    if _pool is None:
        ctx = mp.get_context('spawn')
        # important to use 'spawn', because 'fork' would mean the whole memory is (lazily) copied
        # then due to copy-on-write semantics, it gets duplicated when the parent changes anything
        _pool = ctx.Pool(1, initializer=initialize,
                         initargs=(FLAGS, _original_functions))
    return _pool

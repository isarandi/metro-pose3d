import argparse
import contextlib
import ctypes
import datetime
import functools
import hashlib
import inspect
import itertools
import json
import logging
import multiprocessing as mp
import multiprocessing.connection
import os
import os.path
import pickle
import queue
import signal
import threading
import time
import timeit
import traceback

import numpy as np

TRAIN = 0
VALID = 1
TEST = 2


def cache_result_on_disk(path, forced=None, min_time=None):
    """Helps with caching and restoring the results of a function call on disk.
    Specifically, it returns a function decorator that makes a function cache its result in a file.
    It only evaluates the function once, to generate the cached file. The decorator also adds a
    new keyword argument to the function, called 'forced_cache_update' that can explicitly force
    regeneration of the cached file.

    It has rudimentary handling of arguments by hashing their json representation and appending it
    the hash to the cache filename. This somewhat limited, but is enough for the current uses.

    Set `min_time` to the last significant change to the code within the function.
    If the cached file is older than this `min_time`, the file is regenerated.

    Usage:
        @cache_result_on_disk('/some/path/to/a/file', min_time='2025-12-27T10:12:32')
        def some_function(some_arg):
            ....
            return stuff

    Args:
        path: The path where the function's result is stored.
        forced: do not load from disk, always recreate the cached version
        min_time: recreate cached file if its modification timestamp (mtime) is older than this
           param. The format is like 2025-12-27T10:12:32 (%Y-%m-%dT%H:%M:%S)

    Returns:
        The decorator.
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            inner_forced = forced if forced is not None else kwargs.get('forced_cache_update')
            if 'forced_cache_update' in kwargs:
                del kwargs['forced_cache_update']

            bound_args = inspect.signature(f).bind(*args, **kwargs)
            args_json = json.dumps((bound_args.args, bound_args.kwargs), sort_keys=True)
            hash_string = hashlib.sha1(str(args_json).encode('utf-8')).hexdigest()[:12]

            if args or kwargs:
                noext, ext = os.path.splitext(path)
                suffixed_path = f'{noext}_{hash_string}{ext}'
            else:
                suffixed_path = path

            if not inner_forced and is_file_newer(suffixed_path, min_time):
                logging.debug(f'Loading cached data from {suffixed_path}')
                try:
                    return load_pickle(suffixed_path)
                except Exception as e:
                    print(str(e))
                    logging.error(f'Could not load from {suffixed_path}')
                    raise e

            if os.path.exists(suffixed_path):
                logging.debug(f'Recomputing data for {suffixed_path}')
            else:
                logging.debug(f'Computing data for {suffixed_path}')

            result = f(*args, **kwargs)
            dump_pickle(result, suffixed_path)

            if args or kwargs:
                write_file(args_json, f'{os.path.dirname(path)}/hash_{hash_string}')

            return result

        return wrapped

    return decorator


def timestamp(simplified=False):
    stamp = datetime.datetime.now().isoformat()
    if simplified:
        return stamp.replace(':', '-').replace('.', '-')
    return stamp


def np_all_equal_or_close(a, b, **allclose_options):
    if np.issubdtype(np.result_type(a, b), np.inexact):
        return np.allclose(a, b, **allclose_options)
    else:
        return np.array_equal(a, b)


def print_no_newline(*args, **kwargs):
    print(*args, **kwargs, end='', flush=True)


class FormattableArray:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __format__(self, format_spec):
        # with np.printoptions(
        with numpy_printoptions(
                formatter={'float': lambda x: format(x, format_spec)},
                linewidth=10 ** 6, threshold=10 ** 6):
            return str(self.array)


formattable_array = FormattableArray


@contextlib.contextmanager
def numpy_printoptions(*args, **kwargs):
    original_printoptions = np.get_printoptions()
    try:
        np.set_printoptions(*args, **kwargs)
        yield
    finally:
        np.set_printoptions(**original_printoptions)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(data, file_path, protocol=pickle.HIGHEST_PROTOCOL):
    ensure_path_exists(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol)


def write_file(content, path, is_binary=False):
    mode = 'wb' if is_binary else 'w'
    ensure_path_exists(path)
    with open(path, mode) as f:
        if not is_binary:
            content = str(content)
        f.write(content)
        f.flush()


def ensure_path_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def read_file(path, is_binary=False):
    mode = 'rb' if is_binary else 'r'
    with open(path, mode) as f:
        return f.read()


def split_path(path):
    return os.path.normpath(path).split(os.path.sep)


def last_path_components(path, n_components):
    components = split_path(path)
    return os.path.sep.join(components[-n_components:])


def index_of_first_true(seq, default=None):
    return next((i for i, x in enumerate(seq) if x), default)


def index_of_last_true(seq, default=None):
    i_first = index_of_first_true(seq[::-1])
    if i_first is None:
        return default
    return len(seq) - i_first


def plot_mean_std(ax, x, ys, axis=0):
    mean = np.mean(ys, axis=axis)
    std = np.std(ys, axis=axis)

    ax.plot(x, mean)
    ax.fill_between(x, mean - std, mean + std, alpha=0.3)


def timethem(funcs, number, repeat=3):
    return [np.min(timeit.repeat(func, number=number, repeat=repeat)) / number for func in funcs]


def numpy_concat(a_tuple, axis=0):
    if not a_tuple:
        return np.array([])

    return np.concatenate(a_tuple, axis)


def iterate_repeatedly(seq, shuffle_before_each_epoch=False, rng=None):
    """Iterates over and yields the elements of `iterable` `n_epoch` times.
    if `shuffle_before_each_epoch` is True, the elements are put in a list and shuffled before
    every pass over the data, including the first."""

    if rng is None:
        rng = np.random.RandomState()

    # create a (shallow) copy so shuffling only applies to the copy.
    seq = list(seq)
    for i_epoch in itertools.count():
        logging.debug(f'starting epoch {i_epoch}')
        if shuffle_before_each_epoch:
            logging.debug(f'shuffling {i_epoch}')
            rng.shuffle(seq)
        yield from seq
        logging.debug(f'ended epoch {i_epoch}')


def random_partial_box(random_state):
    def generate():
        x1 = random_state.uniform(0, 0.5)
        x2, y2 = random_state.uniform(0.5, 1, size=2)
        side = x2 - x1
        if not 0.5 < side < y2:
            return None
        return np.array([x1, y2 - side, side, side])

    while True:
        box = generate()
        if box is not None:
            return box


def random_partial_subbox(box, random_state):
    subbox = random_partial_box(random_state)
    topleft = box[:2] + subbox[:2] * box[2:]
    size = subbox[2:] * box[2:]
    return np.concatenate([topleft, size])


def get_all_from_queue(q):
    """Returns all items that are currently in a (queue|multiprocessing).Queue, as a list.
    (Theoretically, if the queue grows faster than it can be read out, this function may never
    return.)

    Args:
        q: The queue.Queue or mp.Queue from which to retrieve the items.
    """

    items = []
    try:
        while True:
            items.append(q.get_nowait())
    except queue.Empty:
        return items


def init_worker_process():
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

    terminate_on_parent_death()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def new_rng(rng):
    if rng is not None:
        return np.random.RandomState(rng.randint(2 ** 32))
    else:
        return np.random.RandomState()


def advance_rng(rng, n_generated_ints):
    for _ in range(n_generated_ints):
        rng.randint(2)


def choice(items, rng):
    return items[rng.randint(len(items))]


def random_uniform_disc(rng):
    """Samples a random 2D point from the unit disc with a uniform distribution."""
    angle = rng.uniform(-np.pi, np.pi)
    radius = np.sqrt(rng.uniform(0, 1))
    return radius * np.array([np.cos(angle), np.sin(angle)])


def init_worker_process_flags(flags):
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
    from init import FLAGS
    for key in flags.__dict__:
        setattr(FLAGS, key, getattr(flags, key))
    import tfu
    tfu.set_data_format(FLAGS.data_format)
    init_worker_process()


def terminate_on_parent_death():
    prctl = ctypes.CDLL("libc.so.6").prctl
    PR_SET_PDEATHSIG = 1
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def safe_subprocess_main_with_flags(flags, func, *args, **kwargs):
    if flags.gui:
        import matplotlib.pyplot as plt
        plt.switch_backend('TkAgg')
    init_worker_process_flags(flags)
    return func(*args, **kwargs)


def reconnecting(fun):
    def wrapped(self, *args, **kwargs):
        while True:
            try:
                return fun(self, *args, **kwargs)
            except (ConnectionResetError, BrokenPipeError, EOFError):
                logging.debug('Reconnecting...')
                self.connect()

    return wrapped


class ReconnectingClient:
    def __init__(self, address):
        self.conn = None
        self.address = address
        logging.debug('Connecting...')
        self.connect()

    def connect(self):
        while True:
            try:
                self.conn = multiprocessing.connection.Client(self.address)
                print('Connected')
                return
            except ConnectionRefusedError:
                logging.debug('Connection refused.')
                time.sleep(1)

    @reconnecting
    def recv(self):
        return self.conn.recv()

    @reconnecting
    def send(self, data):
        return self.conn.send(data)

    @reconnecting
    def communicate(self, data):
        self.conn.send(data)
        return self.conn.recv()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def close(self):
        self.conn.close()


def is_file_newer(path, min_time=None):
    if min_time is None:
        return os.path.exists(path)
    min_time = datetime.datetime.strptime(min_time, '%Y-%m-%dT%H:%M:%S').timestamp()
    return os.path.exists(path) and os.path.getmtime(path) >= min_time


def safe_fun(f, args):
    try:
        return f(*args)
    except BaseException:
        traceback.print_exc()
        raise


class BoundedPool:
    """Wrapper around multiprocessing.Pool that blocks on task submission (`apply_async`) if
    there are already `task_buffer_size` tasks under processing. This can be useful in
    throttling the task producer thread and avoiding too many tasks piling up in the queue and
    eating up too much RAM."""

    def __init__(self, n_processes, task_buffer_size):
        self.pool = mp.Pool(processes=n_processes)
        self.task_semaphore = threading.Semaphore(task_buffer_size)

    def apply_async(self, f, args, callback=None):
        self.task_semaphore.acquire()

        def on_task_completion(result):
            if callback is not None:
                callback(result)
            self.task_semaphore.release()

        self.pool.apply_async(safe_fun, args=(f, args), callback=on_task_completion)

    def close(self):
        self.pool.close()

    def join(self):
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()


def all_disjoint(*seqs):
    union = set()
    for item in itertools.chain(*seqs):
        if item in union:
            return False
        union.add(item)
    return True


def is_running_in_jupyter_notebook():
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def progressbar(*args, **kwargs):
    import tqdm
    import sys
    if is_running_in_jupyter_notebook():
        return tqdm.notebook.tqdm(*args, **kwargs)
    elif sys.stdout.isatty():
        return tqdm.tqdm(*args, dynamic_ncols=True, **kwargs)
    else:
        return args[0]


def ensure_absolute_path(path, root):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(root, path)


def invert_permutation(permutation):
    return np.arange(len(permutation))[np.argsort(permutation)]

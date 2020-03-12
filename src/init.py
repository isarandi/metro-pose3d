#!/usr/bin/env python
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Must import cv2 early
# noinspection PyUnresolvedReferences
import cv2

# Must import TF and slim early due to problem with PyCharm debugger otherwise.
# noinspection PyUnresolvedReferences
import tensorflow as tf

# noinspection PyUnresolvedReferences
import tensorflow.contrib.slim as slim
import argparse
import logging
import os
import shlex
import socket
import sys

import matplotlib.pyplot as plt

import options
import tfu
import util
import paths

FLAGS = argparse.Namespace()

_contexts = []


def initialize():
    global FLAGS
    parse_and_set_global_flags()
    setup_logging()
    logging.info(f'-- Starting --')
    logging.info(f'Host: {socket.gethostname()}')
    logging.info(f'Process id (pid): {os.getpid()}')

    if FLAGS.comment:
        logging.info(f'Comment: {FLAGS.comment}')
    logging.info(f'Raw command: {" ".join(map(shlex.quote, sys.argv))}')
    logging.info(f'Parsed flags: {FLAGS}')
    tfu.set_data_format(FLAGS.data_format)

    if FLAGS.dtype == 'float32':
        tfu.set_dtype(tf.float32)
    elif FLAGS.dtype == 'float16':
        tfu.set_dtype(tf.float16)
    else:
        raise ValueError(f'Training dtype {FLAGS.dtype} not supported, only float16/32.')

    # We parallelize on a coarser level already, openmp just makes things slower
    os.environ['OMP_NUM_THREADS'] = '1'

    # Override the default data format in slim layers
    enter_context(slim.arg_scope(
        [slim.conv2d, slim.conv3d, slim.conv3d_transpose, slim.conv2d_transpose, slim.avg_pool2d,
         slim.separable_conv2d, slim.max_pool2d, slim.batch_norm, slim.spatial_softmax],
        data_format=tfu.data_format()))

    # Override default paddings to SAME
    enter_context(slim.arg_scope([slim.avg_pool2d, slim.max_pool2d], padding='SAME'))

    if FLAGS.gui:
        plt.switch_backend('TkAgg')

    tf.set_random_seed(FLAGS.seed)


def parse_and_set_global_flags():
    global FLAGS
    parser = options.get_parser()
    parser.parse_args(namespace=FLAGS)
    FLAGS.logdir = util.ensure_absolute_path(FLAGS.logdir, root=paths.DATA_ROOT + '/experiments')
    os.makedirs(FLAGS.logdir, exist_ok=True)

    if FLAGS.batch_size_test is None:
        FLAGS.batch_size_test = FLAGS.batch_size

    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = FLAGS.logdir

    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)


def setup_logging():
    global FLAGS

    simple_logfile_path = f'{FLAGS.logdir}/log.txt'
    detailed_logfile_path = f'{FLAGS.logdir}/log_detailed.txt'
    simple_logfile_handler = logging.FileHandler(simple_logfile_path)
    simple_logfile_handler.setLevel(logging.INFO)
    detailed_logfile_handler = logging.FileHandler(detailed_logfile_path)

    simple_formatter = logging.Formatter(
        '{asctime}-{levelname:^1.1} -- {message}', style='{')
    hostname = socket.gethostname().split('.', 1)[0]
    detailed_formatter = logging.Formatter(
        '{asctime} - ' + hostname + ' - {process} - {processName:^12.12} - '
                                    '{threadName:^12.12} - {name:^12.12} - {'
                                    'levelname:^7.7} -- {message}',
        style='{')

    simple_logfile_handler.setFormatter(simple_formatter)
    detailed_logfile_handler.setFormatter(detailed_formatter)
    handlers = [simple_logfile_handler, detailed_logfile_handler]

    if FLAGS.print_log:
        print_handler = logging.StreamHandler(sys.stdout)
        print_handler.setLevel(logging.INFO)
        print_handler.setFormatter(simple_formatter)
        handlers.append(print_handler)
    else:
        # Since we don't want to print the log to stdout, we also redirect stderr to the logfile to
        # save errors for future inspection. But stdout is still stdout.
        sys.stderr.flush()
        new_err_file = open(detailed_logfile_path, 'ab+', 0)
        STDERR_FILENO = 2
        os.dup2(new_err_file.fileno(), STDERR_FILENO)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)


def enter_context(context):
    """Enter a context and keep a reference to it alive."""
    # This fixes the following issue: When `__enter__`ing a local arg_scope variable in a function,
    # the arg_scope seemingly exits when the function returns.
    # This happens because arg_scope is implemented using contextlib.context_manager, so this
    # context manager wraps a generator.
    # However, when the GC cleans up a generator object that hasn't finished iterating yet,
    # it calls the generator's close() method which results in a GeneratorExit exception inside the
    # generator. This can trigger some exception handler or finally block in the generator,
    # which in case of arg_scope basically does the same as an __exit__ (it pops the scope from the
    #  stack)
    context.__enter__()
    global _contexts
    _contexts.append(context)

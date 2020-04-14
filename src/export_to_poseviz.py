#!/usr/bin/env python3

import argparse
import logging
import sys

import numpy as np

import util


def main():
    flags = initialize()
    logging.debug(f'Loading from {flags.in_path}')
    a = np.load(flags.in_path, allow_pickle=True)
    all_results_3d = {}
    for image_path, coords3d_pred in zip(a['image_path'], a['coords3d_pred_world']):
        image_path = image_path.decode('utf8')
        all_results_3d.setdefault(
            image_path, []).append(coords3d_pred.tolist())

    logging.info(f'Writing to file {flags.out_path}')
    util.dump_json(all_results_3d, flags.out_path)


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, default=None)
    parser.add_argument('--loglevel', type=str, default='info')
    flags = parser.parse_args()
    if flags.out_path is None:
        flags.out_path = flags.in_path.replace('.npz', '.json')

    loglevel = dict(error=40, warning=30, info=20, debug=10)[flags.loglevel]
    simple_formatter = logging.Formatter('{asctime}-{levelname:^1.1} -- {message}', style='{')
    print_handler = logging.StreamHandler(sys.stdout)
    print_handler.setLevel(loglevel)
    print_handler.setFormatter(simple_formatter)
    logging.basicConfig(level=loglevel, handlers=[print_handler])

    return flags


if __name__ == '__main__':
    main()

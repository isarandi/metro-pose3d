#!/usr/bin/env python3
import glob
import multiprocessing
import os
import pathlib
import sys

import imageio
import numpy as np


def main():
    if 'DATA_ROOT' not in os.environ:
        print('Set the DATA_ROOT environment variable to the parent dir of the 3dhp directory.')
        sys.exit(1)

    pool = multiprocessing.Pool()
    data_root = os.environ['DATA_ROOT']

    video_paths = glob.glob(f'{data_root}/3dhp/**/imageSequence/*.avi', recursive=True)
    pool.map(extract_frames, video_paths)

    chroma_paths = glob.glob(f'{data_root}/3dhp/**/FGmasks/*.avi', recursive=True)
    pool.map(extract_chroma_masks, chroma_paths)


def extract_chroma_masks(chroma_path):
    """Save a thresholded version of everyt 5th foreground mask."""
    print('Processing', chroma_path)
    video_name = pathlib.Path(chroma_path).stem
    dst_folder_path = pathlib.Path(chroma_path).parents[1] / 'FGmaskImages' / video_name
    os.makedirs(dst_folder_path, exist_ok=True)

    with imageio.get_reader(chroma_path, 'ffmpeg') as reader:
        for i_frame, frame in enumerate(reader):
            if i_frame % 5 == 0:
                dst_filename = f'frame_{i_frame:06d}.png'
                dst_path = os.path.join(dst_folder_path, dst_filename)
                frame = 255 * (frame[..., 0] > 32).astype(np.uint8)
                imageio.imwrite(dst_path, frame)


def extract_frames(src_video_path):
    """Save every 5th frame."""
    print('Processing', src_video_path)
    video_name = pathlib.Path(src_video_path).stem
    dst_folder_path = pathlib.Path(src_video_path).parents[1] / 'Images' / video_name
    os.makedirs(dst_folder_path, exist_ok=True)

    with imageio.get_reader(src_video_path, 'ffmpeg') as reader:
        for i_frame, frame in enumerate(reader):
            if i_frame % 5 == 0:
                dst_filename = f'frame_{i_frame:06d}.jpg'
                dst_path = os.path.join(dst_folder_path, dst_filename)
                imageio.imwrite(dst_path, frame)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse

import numpy as np
import skimage.data
import skimage.transform
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description='MeTRo-Pose3D', allow_abbrev=False)
    parser.add_argument('--model-path', type=str, required=True)
    opts = parser.parse_args()

    images_numpy = np.stack([skimage.transform.resize(skimage.data.astronaut(), (256, 256))])
    images_tensor = tf.convert_to_tensor(images_numpy)
    poses_tensor = estimate_pose(images_tensor, opts.model_path)

    with tf.Session() as sess:
        poses_arr = sess.run(poses_tensor)
        edges = [(1, 0), (0, 18), (0, 2), (2, 3), (3, 4), (0, 8), (8, 9), (9, 10), (18, 5), (5, 6),
                 (6, 7), (18, 11), (11, 12), (12, 13), (15, 14), (14, 1), (17, 16), (16, 1)]
        visualize_pose(image=images_numpy[0], coords=poses_arr[0], edges=edges)


def estimate_pose(im, model_path):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())

    im_t = tf.cast(im, tf.float32)  # / 255 *2 -1
    im_t = tf.transpose(im_t, [0, 3, 1, 2])
    return tf.import_graph_def(
        graph_def, input_map={'input:0': im_t}, return_elements=['pred'])[0].outputs[0]


def visualize_pose(image, coords, edges):
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    # Matplotlib interprets the Z axis as vertical, but our pose
    # has Y as the vertical axis.
    # Therefore we do a 90 degree rotation around the horizontal (X) axis
    coords2 = coords.copy()
    coords[:, 1], coords[:, 2] = coords2[:, 2], -coords2[:, 1]

    fig = plt.figure(figsize=(10, 5))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.set_title('Input')
    image_ax.imshow(image)

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.set_title('Prediction')
    range_ = 800
    pose_ax.set_xlim3d(-range_, range_)
    pose_ax.set_ylim3d(-range_, range_)
    pose_ax.set_zlim3d(-range_, range_)

    for i_start, i_end in edges:
        pose_ax.plot(*zip(coords[i_start], coords[i_end]), marker='o', markersize=2)

    pose_ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

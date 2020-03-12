import tensorflow as tf

import data.datasets
import data.datasets2d
import model.architectures
import tfu
import tfu3d
from init import FLAGS
from model.bone_length_based_backproj import get_bone_lengths, optimize_z_offset_by_bones, \
    optimize_z_offset_by_bones_tensor
from tfu import TRAIN


def build_metro_model(joint_info, learning_phase, t, reuse=None):
    if learning_phase != TRAIN:
        return build_inference_model(joint_info, learning_phase, t, reuse=reuse)

    with tf.name_scope(None, 'Prediction'):
        depth = FLAGS.depth

        def im2pred(im, reuse=reuse):
            net_output = model.architectures.resnet(
                im, n_out=depth * joint_info.n_joints, scope='MainPart', reuse=reuse,
                stride=FLAGS.stride_train, centered_stride=FLAGS.centered_stride,
                resnet_name=FLAGS.architecture)

            net_output_nchw = tfu.std_to_nchw(net_output)
            side = tfu.static_image_shape(net_output)[0]
            reshaped = tf.reshape(net_output_nchw, [-1, depth, joint_info.n_joints, side, side])
            logits = tf.transpose(reshaped, [0, 2, 3, 4, 1])
            softmaxed = tfu.softmax(logits, axis=[2, 3, 4])
            coords3d = tf.stack(tfu.decode_heatmap(softmaxed, [3, 2, 4]), axis=-1)
            return logits, softmaxed, coords3d

        # PREDICT FOR 3D BATCH
        t.logits, t.softmaxed, coords3d = im2pred(t.x)
        t.coords3d_pred = heatmap_to_metric(coords3d, learning_phase)
        if FLAGS.train_mixed:
            coords3d_2d = im2pred(t.x_2d)[-1]
            t.coords3d_pred_2d = heatmap_to_metric(coords3d_2d, learning_phase)

        # PREDICT FOR 2D BATCH
        if FLAGS.train_mixed:
            if FLAGS.dataset == 'mpi_inf_3dhp':
                t.coords3d_pred_2d = adjust_skeleton_3dhp_to_mpii(t.coords3d_pred_2d, joint_info)

            joint_info_2d = data.datasets2d.get_dataset(FLAGS.dataset2d).joint_info
            joint_ids_3d = [joint_info.ids[name] for name in joint_info_2d.names]
            t.coords2d_pred_2d = tf.gather(t.coords3d_pred_2d, joint_ids_3d, axis=1)[..., :2]

        # LOSS 3D BATCH
        t.coords3d_true_rootrel = tfu3d.root_relative(t.coords3d_true)
        t.coords3d_pred_rootrel = tfu3d.root_relative(t.coords3d_pred)

        absdiff = tf.abs(t.coords3d_true_rootrel - t.coords3d_pred_rootrel)
        t.loss3d = tfu.reduce_mean_masked(absdiff, t.joint_validity_mask) / 1000

        # LOSS 2D BATCH
        if FLAGS.train_mixed:
            t.coords2d_true_2d_scaled, t.coords2d_pred_2d = align_2d_skeletons(
                t.coords2d_true_2d, t.coords2d_pred_2d, t.joint_validity_mask_2d)
            scale = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000

            t.loss2d = tfu.reduce_mean_masked(
                tf.abs(t.coords2d_true_2d_scaled - t.coords2d_pred_2d),
                t.joint_validity_mask_2d) * scale
        else:
            t.loss2d = 0

        loss2d_factor = FLAGS.loss2d_factor
        t.loss = t.loss3d + loss2d_factor * t.loss2d

    t.coords3d_pred_orig_cam = to_orig_cam(t.coords3d_pred, t.rot_to_orig_cam, joint_info)
    t.coords3d_true_orig_cam = to_orig_cam(t.coords3d_true, t.rot_to_orig_cam, joint_info)


def build_25d_model(joint_info, learning_phase, t, reuse=None):
    if learning_phase != TRAIN:
        return build_inference_model(joint_info, learning_phase, t, reuse=reuse)

    with tf.name_scope(None, 'Prediction'):
        depth = FLAGS.depth

        def im2pred(im, reuse=reuse):
            net_output = model.architectures.resnet(
                im, n_out=depth * joint_info.n_joints, scope='MainPart', reuse=reuse,
                stride=FLAGS.stride_train, centered_stride=FLAGS.centered_stride,
                resnet_name=FLAGS.architecture)

            side = tfu.static_image_shape(net_output)[0]
            net_output_nchw = tfu.std_to_nchw(net_output)
            reshaped = tf.reshape(net_output_nchw, [-1, depth, joint_info.n_joints, side, side])
            logits = tf.transpose(reshaped, [0, 2, 3, 4, 1])
            softmaxed = tfu.softmax(logits, axis=[2, 3, 4])
            coords3d = tf.stack(tfu.decode_heatmap(softmaxed, [3, 2, 4]), axis=-1)
            return logits, softmaxed, coords3d

        # PREDICT FOR 3D BATCH
        logits, t.softmaxed, coords3d = im2pred(t.x)
        t.coords3d_pred = heatmap_to_25d(coords3d, learning_phase)

        # PREDICT FOR 2D BATCH
        if FLAGS.train_mixed:
            t.coords3d_pred_2d = heatmap_to_25d(im2pred(t.x_2d)[-1], learning_phase)
            if FLAGS.dataset == 'mpi_inf_3dhp':
                t.coords3d_pred_2d = adjust_skeleton_3dhp_to_mpii(t.coords3d_pred_2d, joint_info)

            joint_info_2d = data.datasets2d.get_dataset(FLAGS.dataset2d).joint_info
            joint_ids_3d = [joint_info.ids[name] for name in joint_info_2d.names]
            t.coords2d_pred_2d = tf.gather(t.coords3d_pred_2d, joint_ids_3d, axis=1)[..., :2]

        # LOSS 3D BATCH
        scale = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000
        t.loss2d_3d = tf.reduce_mean(tf.abs(t.coords2d_true - t.coords3d_pred[..., :2])) * scale
        z_ref = t.coords3d_true[..., 2] - t.coords3d_true[:, -1:, 2] + 0.5 * FLAGS.box_size_mm
        t.loss_z = tf.reduce_mean(tf.abs(z_ref - t.coords3d_pred[..., 2])) / 1000

        # LOSS 2D BATCH
        if FLAGS.train_mixed:
            t.loss2d = tfu.reduce_mean_masked(
                tf.abs(t.coords2d_true_2d - t.coords2d_pred_2d), t.joint_validity_mask_2d) * scale
        else:
            t.loss2d = 0

        t.loss3d = (t.loss2d_3d * 2 + t.loss_z) / 3
        t.loss = FLAGS.loss2d_factor * t.loss2d + t.loss3d

        # POST-PROCESSING
        if FLAGS.bone_length_dataset:
            dataset = data.datasets.get_dataset(FLAGS.bone_length_dataset)
        else:
            dataset = data.datasets.current_dataset()

        im_pred2d = t.coords3d_pred[..., :2]
        im_pred2d_homog = to_homogeneous_coords(im_pred2d)
        camcoords2d_homog = matmul_joint_coords(t.inv_intrinsics, im_pred2d_homog)
        delta_z_pred = t.coords3d_pred[..., 2] - t.coords3d_pred[:, -1:, 2]

        if FLAGS.train_on == 'trainval':
            target_bone_lengths = dataset.trainval_bones
        else:
            target_bone_lengths = dataset.train_bones

        z_offset = optimize_z_offset_by_bones(
            camcoords2d_homog, delta_z_pred, target_bone_lengths, joint_info.stick_figure_edges)
        t.coords3d_pred = back_project(camcoords2d_homog, delta_z_pred, z_offset)

    t.coords3d_pred_orig_cam = to_orig_cam(t.coords3d_pred, t.rot_to_orig_cam, joint_info)
    t.coords3d_true_orig_cam = to_orig_cam(t.coords3d_true, t.rot_to_orig_cam, joint_info)


def build_inference_model(joint_info, learning_phase, t, reuse=None):
    stride = FLAGS.stride_train if learning_phase == TRAIN else FLAGS.stride_test

    with tf.name_scope(None, 'Prediction'):
        depth = FLAGS.depth

        def im2pred(im, reuse=reuse):
            net_output = model.architectures.resnet(
                im, n_out=depth * joint_info.n_joints, scope='MainPart', reuse=reuse,
                stride=stride, centered_stride=FLAGS.centered_stride,
                resnet_name=FLAGS.architecture)
            return net_output_to_heatmap_and_coords(net_output, joint_info)

        t.softmaxed, coords3d = im2pred(t.x)
        t.heatmap_pred_z = tf.reduce_sum(t.softmaxed, axis=[2, 3])

        if FLAGS.bone_length_dataset:
            dataset = data.datasets.get_dataset(FLAGS.bone_length_dataset)
        else:
            dataset = data.datasets.current_dataset()
        if 'bone-lengths' in FLAGS.scale_recovery:
            t.coords2d_pred = coords3d[..., :2]
            im_pred2d = heatmap_to_image(t.coords2d_pred, learning_phase)
            im_pred2d_homog = to_homogeneous_coords(im_pred2d)
            camcoords2d_homog = matmul_joint_coords(t.inv_intrinsics, im_pred2d_homog)
            delta_z_pred = (coords3d[..., 2] - coords3d[:, -1:, 2]) * FLAGS.box_size_mm

            target_bone_lengths = (
                dataset.trainval_bones if FLAGS.train_on == 'trainval' else dataset.train_bones)

            if FLAGS.scale_recovery == 'bone-lengths-true':
                bone_lengths_true = get_bone_lengths(t.coords3d_true, joint_info)
                z_offset_gt_bonelen = optimize_z_offset_by_bones_tensor(
                    camcoords2d_homog, delta_z_pred, bone_lengths_true,
                    joint_info.stick_figure_edges)
                t.coords3d_pred = back_project(camcoords2d_homog, delta_z_pred, z_offset_gt_bonelen)
            else:
                z_offset = optimize_z_offset_by_bones(
                    camcoords2d_homog, delta_z_pred, target_bone_lengths,
                    joint_info.stick_figure_edges)
                t.coords3d_pred = back_project(camcoords2d_homog, delta_z_pred, z_offset)
        elif FLAGS.scale_recovery == 'true-root-depth':
            t.coords2d_pred = coords3d[..., :2]
            im_pred2d = heatmap_to_image(t.coords2d_pred, learning_phase)
            im_pred2d_homog = to_homogeneous_coords(im_pred2d)
            camcoords2d_homog = matmul_joint_coords(t.inv_intrinsics, im_pred2d_homog)
            delta_z_pred = (coords3d[..., 2] - coords3d[:, -1:, 2]) * FLAGS.box_size_mm
            t.coords3d_pred = back_project(
                camcoords2d_homog, delta_z_pred, t.coords3d_true[:, -1, 2])
        else:
            t.coords3d_pred = heatmap_to_metric(coords3d, learning_phase)

        t.coords3d_true_rootrel = tfu3d.root_relative(t.coords3d_true)
        t.coords3d_pred_rootrel = tfu3d.root_relative(t.coords3d_pred)

    if learning_phase == TRAIN:
        with tf.name_scope(None, 'Loss'):
            t.loss = tf.reduce_mean(
                tf.abs(t.coords3d_true_rootrel - t.coords3d_pred_rootrel)) / 1000

    t.coords3d_pred_orig_cam = to_orig_cam(t.coords3d_pred, t.rot_to_orig_cam, joint_info)
    t.coords3d_true_orig_cam = to_orig_cam(t.coords3d_true, t.rot_to_orig_cam, joint_info)
    t.coords3d_pred_world = to_orig_cam(
        t.coords3d_pred, t.rot_to_world, joint_info) + tf.expand_dims(t.cam_loc, 1)
    t.coords3d_true_world = to_orig_cam(
        t.coords3d_true, t.rot_to_world, joint_info) + tf.expand_dims(t.cam_loc, 1)


def matmul_joint_coords(matrices, coords):
    return tf.einsum('Bij,BCj->BCi', matrices, coords)


def to_homogeneous_coords(x):
    return tf.concat([x, tf.ones_like(x[..., :1])], axis=-1)


def net_output_to_heatmap_and_coords(net_output, joint_info):
    side = tfu.static_image_shape(net_output)[0]
    depth = int(tfu.static_n_channels(net_output) // joint_info.n_joints)
    net_output_nchw = tfu.std_to_nchw(net_output)
    reshaped = tf.reshape(net_output_nchw, [-1, depth, joint_info.n_joints, side, side])
    transposed = tf.transpose(reshaped, [0, 2, 3, 4, 1])
    softmaxed = tfu.softmax(transposed, axis=[2, 3, 4])
    coords3d = tf.stack(tfu.decode_heatmap(softmaxed, [3, 2, 4]), axis=-1)
    return softmaxed, coords3d


def adjust_skeleton_3dhp_to_mpii(coords3d_pred, joint_info):
    """Move the hips and pelvis towards the neck by a fifth of the pelvis->neck vector
    And move the shoulders up away from the pelvis by 10% of the pelvis->neck vector"""

    j3d = joint_info.ids
    n_joints = joint_info.n_joints
    factor = FLAGS.tdhp_to_mpii_shift_factor
    inverse_factor = -factor / (1 + factor)

    pelvis_neck_vector = coords3d_pred[:, j3d.neck] - coords3d_pred[:, j3d.pelv]
    offset_vector = inverse_factor * pelvis_neck_vector
    n_batch = tfu.dynamic_batch_size(coords3d_pred)

    offsets = []
    for j in range(n_joints):
        if j in (j3d.lhip, j3d.rhip, j3d.pelv):
            offsets.append(offset_vector)
        else:
            offsets.append(tf.zeros([n_batch, 3], dtype=tf.float32))

    offsets = tf.stack([
        tf.zeros([n_batch, 3], dtype=tf.float32)
        if j not in (j3d.lhip, j3d.rhip, j3d.pelv) else offset_vector
        for j in range(n_joints)], axis=1)
    return coords3d_pred + offsets


def align_2d_skeletons(coords_true, coords_pred, joint_validity_mask):
    mean_pred, stdev_pred = tfu.mean_stdev_masked(
        coords_pred, joint_validity_mask, items_axis=1, dimensions_axis=2)

    mean_true, stdev_true = tfu.mean_stdev_masked(
        coords_true, joint_validity_mask, items_axis=1, dimensions_axis=2)

    coords_pred_result = tf.div_no_nan(coords_pred - mean_pred, stdev_pred) * stdev_true
    coords_true_result = coords_true - mean_true
    return coords_true_result, coords_pred_result


def to_orig_cam(x, rot_to_orig_cam, joint_info):
    x = matmul_joint_coords(rot_to_orig_cam, x)
    return tf.where(
        tf.linalg.det(rot_to_orig_cam) > 0, x,
        tf.gather(x, joint_info.mirror_mapping, axis=1))


def back_project(camcoords2d_homog, delta_z, z_offset):
    return camcoords2d_homog * tf.expand_dims(delta_z + tf.expand_dims(z_offset, -1), -1)


def heatmap_to_image(coords, learning_phase):
    stride = FLAGS.stride_train if learning_phase == TRAIN else FLAGS.stride_test
    last_image_pixel = FLAGS.proc_side - 1
    last_receptive_center = last_image_pixel - (last_image_pixel % stride) - 1
    coords_out = coords * last_receptive_center
    if FLAGS.centered_stride:
        coords_out = coords_out + stride // 2
    return coords_out


def heatmap_to_25d(coords, learning_phase):
    coords2d = heatmap_to_image(coords[..., :2], learning_phase)
    return tf.concat([coords2d, coords[:, :, 2:] * FLAGS.box_size_mm], axis=-1)


def heatmap_to_metric(coords, learning_phase):
    coords2d = heatmap_to_image(
        coords[..., :2], learning_phase) * FLAGS.box_size_mm / FLAGS.proc_side
    return tf.concat([coords2d, coords[:, :, 2:] * FLAGS.box_size_mm], axis=-1)

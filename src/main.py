#!/usr/bin/env python3

import init

# This `pass` keeps `init` on top, which needs to be imported first.
pass
import paths
import logging
import os
import os.path
import re

import numpy as np
from attrdict import AttrDict

import data.datasets
import data.datasets2d
import data.data_loading
import helpers

import model.bone_length_based_backproj
import model.volumetric
import session_hooks
import tfasync
import contextlib
import tensorflow as tf
import tfu
import tfu3d
import util
import util3d
from init import FLAGS
from session_hooks import EvaluationMetric
from tfu import TEST, TRAIN, VALID
import attrdict


def main():
    print('Starting', flush=True)
    init.initialize()
    if FLAGS.train:
        train()
    if FLAGS.test:
        test()
    if FLAGS.export_file:
        export()


def train():
    logging.info('Training phase.')
    rng = np.random.RandomState(FLAGS.seed)
    n_completed_steps = get_number_of_already_completed_steps(FLAGS.logdir)

    t_train = build_graph(
        TRAIN, rng=util.new_rng(rng), n_epochs=FLAGS.epochs, n_completed_steps=n_completed_steps)
    logging.info(f'Number of trainable parameters: {tfu.count_trainable_params():,}')
    t_valid = (build_graph(VALID, shuffle=True, rng=util.new_rng(rng))
               if FLAGS.validate_period else None)

    helpers.run_train_loop(
        t_train.train_op, checkpoint_dir=FLAGS.checkpoint_dir, load_path=FLAGS.load_path,
        hooks=make_training_hooks(t_train, t_valid), init_fn=get_init_fn())
    logging.info('Ended training.')


def test():
    logging.info('Test (eval) phase.')
    tf.reset_default_graph()
    n_epochs = FLAGS.epochs if FLAGS.occlude_test or FLAGS.test_aug or FLAGS.multiepoch_test else 1
    t = build_graph(TEST, n_epochs=n_epochs, shuffle=False)

    test_counter = tfu.get_or_create_counter('testc')
    counter_hook = session_hooks.counter_hook(test_counter)
    example_hook = session_hooks.log_increment_per_sec(
        'Examples', test_counter.var * FLAGS.batch_size_test, None, every_n_secs=FLAGS.hook_seconds)

    hooks = [example_hook, counter_hook]
    dataset = data.datasets.current_dataset()
    if FLAGS.gui:
        plot_hook = session_hooks.send_to_worker_hook(
            [(t.x[0] + 1) / 2, t.coords3d_pred[0], t.coords3d_true[0]],
            util3d.plot3d_worker,
            worker_args=[dataset.joint_info.stick_figure_edges],
            worker_kwargs=dict(batched=False, interval=100, has_ground_truth=True),
            every_n_steps=1, use_threading=False)
        rate_limit_hook = session_hooks.rate_limit_hook(0.5)
        hooks.append(plot_hook)
        hooks.append(rate_limit_hook)

    fetch_names = [
        'image_path', 'coords3d_true_orig_cam', 'coords3d_pred_orig_cam', 'coords3d_true_world',
        'coords3d_pred_world', 'activity_name', 'scene_name',
        'joint_validity_mask', 'confidences', 'coords3d_pred_backproj_orig_cam', 'compactness']
    fetch_tensors = {fetch_name: t[fetch_name] for fetch_name in fetch_names}

    global_init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()

    def init_fn(_, sess):
        sess.run([global_init_op, local_init_op, test_counter.reset_op])

    f = helpers.run_eval_loop(
        fetches_to_collect=fetch_tensors, load_path=FLAGS.load_path,
        checkpoint_dir=FLAGS.checkpoint_dir, hooks=hooks, init_fn=init_fn)
    save_results(f)


def export():
    from tensorflow.tools.graph_transforms import TransformGraph
    t = attrdict.AttrDict()
    t.x = tf.placeholder(
        shape=[None, FLAGS.proc_side, FLAGS.proc_side, 3], dtype=tf.float32, name='input')
    t.x = tfu.nhwc_to_std(t.x)
    joint_info = data.datasets.current_dataset().joint_info
    model.volumetric.build_inference_model(joint_info, TEST, t)

    # Convert to the original joint order as defined in the original datasets
    # (i.e. put the pelvis back to its place from the last position,
    # because this codebase normally uses the last position for the pelvis in all cases for
    # consistency)
    if FLAGS.dataset == 'merged':
        permutation = [0, 1, 18, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    elif FLAGS.dataset == 'h36m':
        permutation = [16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    else:
        assert FLAGS.dataset == 'mpi_inf_3dhp'
        permutation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 14, 15]

    tf.gather(t.coords3d_pred_rootrel, permutation, axis=1, name='output')
    joint_info = joint_info.permute_joints(permutation)

    if FLAGS.load_path:
        if not os.path.isabs(FLAGS.load_path) and FLAGS.checkpoint_dir:
            load_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.load_path)
        else:
            load_path = FLAGS.load_path
    else:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        load_path = checkpoint.model_checkpoint_path
    checkpoint_dir = os.path.dirname(load_path)

    tf.convert_to_tensor(np.array(joint_info.names), name='joint_names')
    tf.convert_to_tensor(np.array(joint_info.stick_figure_edges), name='joint_edges')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, load_path)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(),
            ['output', 'joint_names', 'joint_edges'])

        transforms = [
            'merge_duplicate_nodes',
            'strip_unused_nodes',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms']

        optimized_graph_def = TransformGraph(
            frozen_graph_def, [], ['output', 'joint_names', 'joint_edges'], transforms)
        tf.train.write_graph(
            optimized_graph_def, logdir=checkpoint_dir, as_text=False,
            name=FLAGS.export_file)
        logging.info(f'Exported the model to {checkpoint_dir}/{FLAGS.export_file}')


def save_results(f):
    default_path = f'results_{util.timestamp()}.npz'
    result_path = FLAGS.result_path if FLAGS.result_path else default_path
    if not os.path.isabs(result_path):
        result_path = os.path.join(FLAGS.logdir, result_path)
    ordered_indices = np.argsort(f.image_path)
    util.ensure_path_exists(result_path)
    logging.info(f'Saving results to {result_path}')
    np.savez(
        result_path,
        image_path=f.image_path[ordered_indices],
        coords3d_true=f.coords3d_true_orig_cam[ordered_indices],
        coords3d_pred=f.coords3d_pred_orig_cam[ordered_indices],
        coords3d_true_world=f.coords3d_true_world[ordered_indices],
        coords3d_pred_world=f.coords3d_pred_world[ordered_indices],
        coords3d_pred_backproj=f.coords3d_pred_backproj_orig_cam[ordered_indices],
        activity_name=f.activity_name[ordered_indices],
        scene_name=f.scene_name[ordered_indices],
        joint_validity_mask=f.joint_validity_mask[ordered_indices],
    )


def build_graph(
        learning_phase, reuse=tf.AUTO_REUSE, n_epochs=None, shuffle=None, drop_remainder=None,
        rng=None, n_completed_steps=0):
    tfu.set_is_training(learning_phase == TRAIN)

    dataset3d = data.datasets.current_dataset()
    if FLAGS.train_mixed:
        dataset2d = data.datasets2d.get_dataset(FLAGS.dataset2d)
    else:
        dataset2d = None

    t = AttrDict()
    t.global_step = tf.train.get_or_create_global_step()

    examples = helpers.get_examples(dataset3d, learning_phase, FLAGS)
    phase_name = tfu.PHASE_NAME[learning_phase]
    t.n_examples = len(examples)
    logging.info(f'Number of {phase_name} examples: {t.n_examples:,}')

    if n_epochs is None:
        n_total_steps = None
    else:
        batch_size = FLAGS.batch_size if learning_phase == TRAIN else FLAGS.batch_size_test
        n_total_steps = (len(examples) * n_epochs) // batch_size

    if rng is None:
        rng = np.random.RandomState()

    rng_2d = util.new_rng(rng)
    rng_3d = util.new_rng(rng)

    @contextlib.contextmanager
    def empty_context():
        yield

    name_scope = tf.name_scope(None, 'training') if learning_phase == TRAIN else empty_context()
    with name_scope:
        if learning_phase == TRAIN and FLAGS.train_mixed:
            examples2d = [*dataset2d.examples[tfu.TRAIN], *dataset2d.examples[tfu.VALID]]
            build_mixed_batch(
                t, dataset3d, dataset2d, examples, examples2d, learning_phase,
                batch_size3d=FLAGS.batch_size, batch_size2d=FLAGS.batch_size_2d,
                shuffle=shuffle, rng_2d=rng_2d, rng_3d=rng_3d, max_unconsumed=FLAGS.max_unconsumed,
                n_completed_steps=n_completed_steps, n_total_steps=n_total_steps)
        elif FLAGS.multiepoch_test:
            batch_size = FLAGS.batch_size if learning_phase == TRAIN else FLAGS.batch_size_test
            helpers.build_input_batch(
                t, examples, data.data_loading.load_and_transform3d,
                (dataset3d.joint_info, learning_phase), learning_phase, batch_size,
                FLAGS.workers, shuffle=shuffle, drop_remainder=drop_remainder, rng=rng_3d,
                max_unconsumed=FLAGS.max_unconsumed, n_completed_steps=n_completed_steps,
                n_total_steps=n_total_steps, n_test_epochs=FLAGS.epochs)
            (t.image_path, t.x, t.coords3d_true, t.coords2d_true, t.inv_intrinsics,
             t.rot_to_orig_cam, t.rot_to_world, t.cam_loc, t.joint_validity_mask,
             t.is_joint_in_fov, t.activity_name, t.scene_name) = t.batch
        else:
            batch_size = FLAGS.batch_size if learning_phase == TRAIN else FLAGS.batch_size_test
            helpers.build_input_batch(
                t, examples, data.data_loading.load_and_transform3d,
                (dataset3d.joint_info, learning_phase), learning_phase, batch_size,
                FLAGS.workers, shuffle=shuffle, drop_remainder=drop_remainder, rng=rng_3d,
                max_unconsumed=FLAGS.max_unconsumed, n_completed_steps=n_completed_steps,
                n_total_steps=n_total_steps)
            (t.image_path, t.x, t.coords3d_true, t.coords2d_true, t.inv_intrinsics,
             t.rot_to_orig_cam, t.rot_to_world, t.cam_loc, t.joint_validity_mask,
             t.is_joint_in_fov, t.activity_name, t.scene_name) = t.batch

        if FLAGS.scale_recovery == 'metro':
            model.volumetric.build_metro_model(
                dataset3d.joint_info, learning_phase, t, reuse=reuse)
        else:
            model.volumetric.build_25d_model(
                dataset3d.joint_info, learning_phase, t, reuse=reuse)

        if 'coords3d_true' in t:
            build_eval_metrics(t)

        if learning_phase == TRAIN:
            build_train_op(t)
            build_summaries(t)
        return t


@tfu.in_name_scope('InputPipeline')
def build_mixed_batch(
        t, dataset3d, dataset2d, examples3d, examples2d, learning_phase,
        batch_size3d=None, batch_size2d=None, shuffle=None, rng_2d=None, rng_3d=None,
        max_unconsumed=256, n_completed_steps=0, n_total_steps=None):
    if shuffle is None:
        shuffle = learning_phase == TRAIN

    (t.image_path_2d, t.x_2d, t.coords2d_true_2d,
     t.joint_validity_mask_2d) = helpers.build_input_batch(
        t, examples2d, data.data_loading.load_and_transform2d,
        (dataset2d.joint_info, learning_phase), learning_phase, batch_size2d, FLAGS.workers,
        shuffle=shuffle, rng=rng_2d, max_unconsumed=max_unconsumed,
        n_completed_steps=n_completed_steps, n_total_steps=n_total_steps)

    (t.image_path, t.x, t.coords3d_true, t.coords2d_true, t.inv_intrinsics,
     t.rot_to_orig_cam, t.rot_to_world, t.cam_loc, t.joint_validity_mask,
     t.is_joint_in_fov, t.activity_name, t.scene_name) = helpers.build_input_batch(
        t, examples3d, data.data_loading.load_and_transform3d,
        (dataset3d.joint_info, learning_phase), learning_phase, batch_size3d, FLAGS.workers,
        shuffle=shuffle, rng=rng_3d, max_unconsumed=max_unconsumed,
        n_completed_steps=n_completed_steps, n_total_steps=n_total_steps)


@tfu.in_name_scope('Optimizer')
def build_train_op(t):
    t.global_step = tf.train.get_or_create_global_step()
    t.learning_rate = learning_rate_schedule(
        t.global_step, t.n_examples / FLAGS.batch_size, FLAGS.learning_rate, FLAGS)

    n_steps_total = t.n_examples / FLAGS.batch_size * FLAGS.epochs
    weight_decay = (FLAGS.weight_decay * (
            t.learning_rate / FLAGS.learning_rate) / n_steps_total ** 0.5)
    t.optimizer = tf.contrib.opt.AdamWOptimizer(
        weight_decay=weight_decay, learning_rate=t.learning_rate, epsilon=FLAGS.epsilon)

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='training')

    def minimize(optimizer, loss, var_list, update_list, step=t.global_step):
        gradients = tfu.gradients_with_loss_scaling(loss, var_list, 256)
        gradients_and_vars = list(zip(gradients, var_list))
        m = optimizer.apply_gradients(gradients_and_vars, global_step=step)
        return tf.group(m, *update_list)

    t.train_op = minimize(t.optimizer, t.loss, trainable_vars, update_ops)


def learning_rate_schedule(global_step, steps_per_epoch, base_learning_rate, flags):
    phase1_steps = 25 * steps_per_epoch
    phase2_steps = 2 * steps_per_epoch
    global_step_float = tf.cast(global_step, tf.float32)

    if flags.stretch_schedule:
        phase1_steps = int(flags.stretch_schedule * phase1_steps)
        phase2_steps = int(flags.stretch_schedule2 * phase2_steps)

    b = tf.cast(base_learning_rate, tf.float32)

    phase1_lr = tf.train.exponential_decay(
        b, global_step=global_step_float, decay_rate=1 / 3,
        decay_steps=phase1_steps, staircase=False)

    phase2_lr = tf.train.exponential_decay(
        b * tf.cast(1 / 30, tf.float32),
        global_step=global_step_float - phase1_steps,
        decay_rate=0.3, decay_steps=phase2_steps, staircase=False)

    return tf.where(global_step_float < phase1_steps, phase1_lr, phase2_lr)


@tfu.in_name_scope('EvalMetrics')
def build_eval_metrics(t):
    rootrelative_diff = tfu3d.root_relative(t.coords3d_pred - t.coords3d_true)
    dist = tf.norm(rootrelative_diff, axis=-1)
    t.mean_error = tfu.reduce_mean_masked(dist, t.joint_validity_mask)
    t.coords3d_pred_procrustes = tfu3d.rigid_align(
        t.coords3d_pred, t.coords3d_true,
        joint_validity_mask=t.joint_validity_mask, scale_align=True)

    rootrelative_diff_procrust = tfu3d.root_relative(t.coords3d_pred_procrustes - t.coords3d_true)
    dist_procrustes = tf.norm(rootrelative_diff_procrust, axis=-1)
    t.mean_error_procrustes = tfu.reduce_mean_masked(dist_procrustes, t.joint_validity_mask)

    threshold = np.float32(150)
    auc_score = tf.maximum(np.float32(0), 1 - dist / threshold)
    t.auc = tfu.reduce_mean_masked(auc_score, t.joint_validity_mask, axis=0)
    t.mean_auc = tfu.reduce_mean_masked(auc_score, t.joint_validity_mask)

    is_correct = tf.cast(dist <= threshold, tf.float32)
    t.pck = tfu.reduce_mean_masked(is_correct, t.joint_validity_mask, axis=0)
    t.mean_pck = tfu.reduce_mean_masked(is_correct, t.joint_validity_mask)


def build_summaries(t):
    t.epoch = t.global_step // (t.n_examples // FLAGS.batch_size)
    opnames = 'epoch,learning_rate,mean_error,mean_error_procrustes,loss,loss2d,loss3d'.split(',')
    scalar_ops = ({name: t[name] for name in opnames if name in t})
    summaries = [tf.summary.scalar(name, op, collections=[]) for name, op in scalar_ops.items()]
    t.summary_op = tf.summary.merge(summaries)


def make_training_hooks(t_train, t_valid):
    saver = tf.train.Saver(max_to_keep=2, save_relative_paths=True)

    checkpoint_state = tf.train.get_checkpoint_state(FLAGS.logdir)
    if checkpoint_state:
        saver.recover_last_checkpoints(checkpoint_state.all_model_checkpoint_paths)

    global_step_tensor = tf.train.get_or_create_global_step()
    checkpoint_hook = tf.train.CheckpointSaverHook(FLAGS.logdir, saver=saver, save_secs=30 * 60)
    total_batch_size = FLAGS.batch_size * (2 if FLAGS.train_mixed else 1)
    example_counter_hook = session_hooks.log_increment_per_sec(
        'Training images', t_train.global_step * total_batch_size, every_n_secs=FLAGS.hook_seconds,
        summary_output_dir=FLAGS.logdir)

    i_epoch = t_train.global_step // (t_train.n_examples // FLAGS.batch_size)
    logger_hook = session_hooks.logger_hook(
        'Epoch {:03d}, global step {:07d}. Loss: {:.5e}',
        [i_epoch, t_train.global_step, t_train.loss],
        every_n_secs=FLAGS.hook_seconds)

    hooks = [example_counter_hook, logger_hook, checkpoint_hook]

    if FLAGS.epochs:
        last_step = ((t_train.n_examples * FLAGS.epochs) // FLAGS.batch_size) - 1
        eta_hook = session_hooks.eta_hook(
            last_step + 1, every_n_secs=600, summary_output_dir=FLAGS.logdir)
        hooks.append(eta_hook)

    if FLAGS.validate_period:
        every_n_steps = (
            int(np.round(FLAGS.validate_period * (t_train.n_examples // FLAGS.batch_size))))

        max_valid_steps = np.ceil(t_valid.n_examples / FLAGS.batch_size_test)
        summary_output_dir = FLAGS.logdir if FLAGS.tensorboard else None

        metrics = [
            EvaluationMetric(t_valid.mean_error, 'MPJPE', '.3f', is_higher_better=False),
            EvaluationMetric(t_valid.mean_error_procrustes, 'MPJPE-procrustes', '.3f',
                             is_higher_better=False),
            EvaluationMetric(t_valid.mean_pck, '3DPCK@150mm', '.3%'),
            EvaluationMetric(t_valid.mean_auc, 'AUC', '.3%'),
        ]

        validation_hook = session_hooks.validation_hook(
            metrics, summary_output_dir=summary_output_dir, max_steps=max_valid_steps,
            max_seconds=120, every_n_steps=every_n_steps, _step_tensor=global_step_tensor)
        hooks.append(validation_hook)

    if FLAGS.tensorboard:
        other_summary_ops = [a for a in [tf.summary.merge_all()] if a is not None]
        summary_hook = tf.train.SummarySaverHook(
            save_steps=1, output_dir=FLAGS.logdir,
            summary_op=tf.summary.merge([*other_summary_ops, t_train.summary_op]))
        summary_hook = tfasync._PeriodicHookWrapper(
            summary_hook, every_n_steps=10, step_tensor=global_step_tensor)
        hooks.append(summary_hook)

    if FLAGS.gui:
        dataset = data.datasets.current_dataset()
        plot_hook = session_hooks.send_to_worker_hook(
            [(t_train.x[0] + 1) / 2, t_train.coords3d_pred[0], t_train.coords3d_true[0]],
            util3d.plot3d_worker, worker_args=[dataset.joint_info.stick_figure_edges],
            worker_kwargs=dict(batched=False, interval=100), every_n_secs=FLAGS.hook_seconds,
            use_threading=False)
        hooks.append(plot_hook)
        if 'coords3d_pred_2d' in t_train:
            plot_hook = session_hooks.send_to_worker_hook(
                [(t_train.x_2d[0] + 1) / 2, t_train.coords3d_pred_2d[0],
                 t_train.coords3d_pred_2d[0]],
                util3d.plot3d_worker, worker_args=[dataset.joint_info.stick_figure_edges],
                worker_kwargs=dict(batched=False, interval=100, has_ground_truth=False),
                every_n_secs=FLAGS.hook_seconds,
                use_threading=False)
            hooks.append(plot_hook)

    return hooks


def get_init_fn():
    if FLAGS.init == 'scratch':
        return None
    elif FLAGS.init != 'pretrained':
        raise NotImplementedError

    if FLAGS.architecture.startswith('resnet_v2'):
        default_weight_subpath = f'{FLAGS.architecture}_2017_04_14/{FLAGS.architecture}.ckpt'
    elif FLAGS.architecture.startswith('mobilenet'):
        default_weight_subpath = f'v3-large_224_1.0_float/pristine/model.ckpt-540000'
    else:
        raise Exception(
            f'No default pretrained weights configured for architecture {FLAGS.architecture}')

    if FLAGS.init_path:
        weight_path = FLAGS.init_path
        checkpoint_scope = f'MainPart/{FLAGS.architecture}'
    else:
        weight_path = f'{paths.DATA_ROOT}/pretrained/{default_weight_subpath}'
        if not os.path.exists(weight_path) and not os.path.exists(weight_path + '.index'):
            download_pretrained_weights()
        checkpoint_scope = FLAGS.architecture

    loaded_scope = f'MainPart/{FLAGS.architecture}'
    do_not_load = ['Adam', 'Momentum', 'noload']
    if FLAGS.init_logits_random:
        do_not_load.append('logits')

    return tfu.make_pretrained_weight_loader(
        weight_path, loaded_scope, checkpoint_scope, do_not_load)


def download_pretrained_weights():
    import urllib.request
    import tarfile

    logging.info(f'Downloading ImageNet pretrained weights for {FLAGS.architecture}')
    filename = f'{FLAGS.architecture}_2017_04_14.tar.gz'
    target_path = f'{paths.DATA_ROOT}/pretrained/{FLAGS.architecture}_2017_04_14/{filename}'
    util.ensure_path_exists(target_path)
    urllib.request.urlretrieve(f'http://download.tensorflow.org/models/{filename}', target_path)
    with tarfile.open(target_path) as f:
        f.extractall(f'{paths.DATA_ROOT}/pretrained/{FLAGS.architecture}_2017_04_14')
    os.remove(target_path)


def get_number_of_already_completed_steps(logdir):
    """Find out how many training steps have already been completed in case we are resuming."""
    if os.path.exists(f'{logdir}/checkpoint'):
        text = util.read_file(f'{logdir}/checkpoint')
        return int(re.search('model_checkpoint_path: "model.ckpt-(?P<num>\d+)"', text)['num'])
    else:
        return 0


if __name__ == '__main__':
    main()

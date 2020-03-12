import numpy as np

import tfu3d
import util3d


def get_pck(rel_dists):
    return np.mean(rel_dists <= 1)


def get_auc(rel_dists):
    return np.mean(np.maximum(0, 1 - rel_dists))


def h36m_numbers(
        coords3d_true, coords3d_pred, activity_name, procrustes=False, joint_validity_mask=None):
    if joint_validity_mask is None:
        joint_validity_mask = np.full_like(coords3d_true[..., 0], fill_value=True, dtype=np.bool)

    coords3d_true = tfu3d.root_relative(coords3d_true)
    coords3d_pred = tfu3d.root_relative(coords3d_pred)

    if procrustes in ('rigid', 'rigid+scale'):
        coords3d_pred = util3d.rigid_align_many(
            coords3d_pred, coords3d_true, joint_validity_mask=joint_validity_mask,
            scale_align=procrustes == 'rigid+scale')

    dist = np.linalg.norm(coords3d_true - coords3d_pred, axis=-1)
    overall_mean_error = np.mean(dist[joint_validity_mask])

    result = []
    ordered_actions = 'Directions,Discussion,Eating,Greeting,Phoning,Posing,Purchases,Sitting,' \
                      'SittingDown,Smoking,Photo,Waiting,Walking,WalkDog,WalkTogether'.split(',')
    for activity in ordered_actions:
        activity = activity.encode('utf8')
        mask = np.logical_and(np.expand_dims(activity_name == activity, -1), joint_validity_mask)
        act_mean_error = np.mean(dist[mask])
        result.append(act_mean_error)

    result.append(overall_mean_error)
    return result


def tdhp_numbers(
        coords3d_true, coords3d_pred, activity_name, scene_name, procrustes=False,
        joint_validity_mask=None):
    if joint_validity_mask is None:
        joint_validity_mask = np.full_like(coords3d_true[..., 0], fill_value=True, dtype=np.bool)

    coords3d_true = tfu3d.root_relative(coords3d_true)
    coords3d_pred = tfu3d.root_relative(coords3d_pred)

    if procrustes in ('rigid', 'rigid+scale'):
        coords3d_pred = util3d.rigid_align_many(
            coords3d_pred, coords3d_true, joint_validity_mask=joint_validity_mask,
            scale_align=procrustes == 'rigid+scale')

    dist = np.linalg.norm(coords3d_true - coords3d_pred, axis=-1)
    overall_mean_error = np.mean(dist[joint_validity_mask])

    ordered_actions = ('Stand/Walk,Exercise,Sit on Chair,'
                       'Reach/Crouch,On Floor,Sports,Misc.'.split(','))
    ordered_scenes = ['green-screen', 'no-green-screen', 'outdoor']

    rel_dist = dist / 150
    overall_pck = get_pck(rel_dist[joint_validity_mask])
    overall_auc = get_auc(rel_dist[joint_validity_mask])

    result = []
    for activity in ordered_actions:
        activity = activity.encode('utf8')
        mask = np.logical_and(np.expand_dims(activity_name == activity, -1), joint_validity_mask)
        act_pck = get_pck(rel_dist[mask])
        result.append(act_pck)

    for scene in ordered_scenes:
        scene = scene.encode('utf8')
        mask = np.logical_and(np.expand_dims(scene_name == scene, -1), joint_validity_mask)
        act_pck = get_pck(rel_dist[mask])
        result.append(act_pck)

    result += [overall_pck, overall_auc, overall_mean_error]
    return result

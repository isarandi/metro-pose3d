import functools
import itertools
import logging

import more_itertools
import numpy as np
from attrdict import AttrDict

import util

TRAIN = 0
VALID = 1
TEST = 2


class Pose3DDataset:
    def __init__(
            self, joint_info, train_examples=None, valid_examples=None, test_examples=None):
        self.joint_info = joint_info
        self.examples = {
            TRAIN: train_examples or [],
            VALID: valid_examples or [],
            TEST: test_examples or []}

        trainval_examples = [*self.examples[TRAIN], *self.examples[VALID]]
        if trainval_examples:
            self.trainval_bones = self.get_mean_bones(trainval_examples)
        if self.examples[TRAIN]:
            self.train_bones = self.get_mean_bones(self.examples[TRAIN])

    def get_mean_bones(self, examples):
        coords3d = np.stack([ex.world_coords for ex in examples], axis=0)
        return [
            np.nanmean(np.linalg.norm(coords3d[:, i] - coords3d[:, j], axis=-1))
            for i, j in self.joint_info.stick_figure_edges]


class Pose3DExample:
    def __init__(
            self, image_path, world_coords, bbox, camera, *,
            activity_name='unknown', scene_name='unknown', mask=None, univ_coords=None):
        self.image_path = image_path
        self.world_coords = world_coords
        self.univ_coords = univ_coords if univ_coords is not None else None
        self.bbox = np.asarray(bbox)
        self.camera = camera
        self.activity_name = activity_name
        self.scene_name = scene_name
        self.mask = mask


class JointInfo:
    def __init__(self, joints, edges):
        if isinstance(joints, dict):
            self.ids = joints
        elif isinstance(joints, (list, tuple)):
            self.ids = JointInfo.make_id_map(joints)
        elif isinstance(joints, str):
            self.ids = JointInfo.make_id_map(joints.split(','))
        else:
            raise Exception

        self.names = list(sorted(self.ids.keys(), key=self.ids.get))
        self.n_joints = len(self.ids)

        if isinstance(edges, str):
            self.stick_figure_edges = []
            for path_str in edges.split(','):
                joint_names = path_str.split('-')
                for joint_name1, joint_name2 in more_itertools.pairwise(joint_names):
                    if joint_name1 in self.ids and joint_name2 in self.ids:
                        edge = (self.ids[joint_name1], self.ids[joint_name2])
                        self.stick_figure_edges.append(edge)
        else:
            self.stick_figure_edges = edges

        # the index of the joint on the opposite side (e.g. maps index of left wrist to index
        # of right wrist)
        self.mirror_mapping = [
            self.ids[JointInfo.other_side_joint_name(name)] for name in self.names]

    def update_names(self, new_names):
        if isinstance(new_names, str):
            new_names = new_names.split(',')

        self.names = new_names
        new_ids = AttrDict()
        for i, new_name in enumerate(new_names):
            new_ids[new_name] = i
        self.ids = new_ids

    @staticmethod
    def make_id_map(names):
        return AttrDict(dict(zip(names, itertools.count())))

    @staticmethod
    def other_side_joint_name(name):
        if name.startswith('l'):
            return 'r' + name[1:]
        elif name.startswith('r'):
            return 'l' + name[1:]
        else:
            return name

    def permute_joints(self, permutation):
        inv_perm = util.invert_permutation(permutation)
        new_names = [self.names[x] for x in permutation]
        new_edges = [(inv_perm[i], inv_perm[j]) for i, j in self.stick_figure_edges]
        return JointInfo(new_names, new_edges)


def make_h36m_incorrect_S9(*args, **kwargs):
    import data.h36m
    return data.h36m.make_h36m(*args, **kwargs, correct_S9=False)


def make_h36m(*args, **kwargs):
    import data.h36m
    return data.h36m.make_h36m(*args, **kwargs)


def make_h36m_partial(*args, **kwargs):
    import data.h36m
    return data.h36m.make_h36m(*args, **kwargs, partial_visibility=True)


def make_mpi_inf_3dhp():
    import data.mpi_inf_3dhp
    return data.mpi_inf_3dhp.make_mpi_inf_3dhp()


def make_mpi_inf_3dhp_correctedTS6():
    import data.mpi_inf_3dhp
    return data.mpi_inf_3dhp.make_mpi_inf_3dhp(ts6_corr=True)


def current_dataset():
    from init import FLAGS
    return get_dataset(FLAGS.dataset)


def make_merged():
    joint_names = ['neck', 'nose', 'lsho', 'lelb', 'lwri', 'lhip', 'lkne', 'lank', 'rsho', 'relb',
                   'rwri', 'rhip', 'rkne', 'rank', 'leye', 'lear', 'reye', 'rear', 'pelv',
                   'htop_tdhp', 'neck_tdhp', 'rsho_tdhp', 'lsho_tdhp', 'rhip_tdhp', 'lhip_tdhp',
                   'spin_tdhp', 'head_tdhp', 'pelv_tdhp', 'rhip_h36m', 'lhip_h36m', 'tors_h36m',
                   'neck_h36m', 'head_h36m', 'htop_h36m', 'lsho_h36m', 'rsho_h36m', 'pelv_h36m',
                   'lhip_tdpw', 'rhip_tdpw', 'bell_tdpw', 'che1_tdpw', 'che2_tdpw', 'ltoe_tdpw',
                   'rtoe_tdpw', 'neck_tdpw', 'lcla_tdpw', 'rcla_tdpw', 'head_tdpw', 'lsho_tdpw',
                   'rsho_tdpw', 'lhan_tdpw', 'rhan_tdpw', 'pelv_tdpw']
    edges = [(1, 0), (0, 18), (0, 2), (2, 3), (3, 4), (0, 8), (8, 9), (9, 10), (18, 5), (5, 6),
             (6, 7), (18, 11), (11, 12), (12, 13), (15, 14), (14, 1), (17, 16), (16, 1)]
    joint_info = JointInfo(joint_names, edges)
    return Pose3DDataset(joint_info)


@functools.lru_cache()
def get_dataset(dataset_name):
    from init import FLAGS

    if dataset_name.endswith('.pkl'):
        return util.load_pickle(util.ensure_absolute_path(dataset_name))
    logging.debug(f'Making dataset {dataset_name}...')

    kwargs = {}

    def string_to_intlist(string):
        return tuple(int(s) for s in string.split(','))

    for subj_key in ['train_subjects', 'valid_subjects', 'test_subjects']:
        if hasattr(FLAGS, subj_key) and getattr(FLAGS, subj_key):
            kwargs[subj_key] = string_to_intlist(getattr(FLAGS, subj_key))

    return globals()[f'make_{dataset_name}'](**kwargs)

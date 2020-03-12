import functools
import logging
import os
import os.path

import numpy as np

import boxlib
import cameralib
import imageio
import improc
import matlabfile
import paths
import util
from data.datasets import JointInfo
from tfu import TEST, TRAIN, VALID


class Pose2DDataset:
    def __init__(
            self, joint_info, train_examples=None, valid_examples=None, test_examples=None):
        self.joint_info = joint_info
        self.examples = {
            TRAIN: train_examples or [], VALID: valid_examples or [], TEST: test_examples or []}


class Pose2DExample:
    def __init__(self, image_path, coords, bbox=None):
        self.image_path = image_path
        self.coords = coords
        self.bbox = bbox


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/mpii.pkl', min_time="2020-03-03T12:49:04")
def make_mpii():
    joint_names = 'rank,rkne,rhip,lhip,lkne,lank,pelv,thor,neck,head,rwri,relb,rsho,lsho,lelb,lwri'
    edges = 'lsho-lelb-lwri,rsho-relb-rwri,lhip-lkne-lank,rhip-rkne-rank,neck-head,pelv-thor'
    joint_info_full = JointInfo(joint_names, edges)

    joint_names_used = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'
    joint_info_used = JointInfo(joint_names_used, edges)
    dataset = Pose2DDataset(joint_info_used)
    selected_joints = [joint_info_full.ids[name] for name in joint_info_used.names]

    mat_path = f'{paths.DATA_ROOT}/mpii/mpii_human_pose_v1_u12_1.mat'
    s = matlabfile.load(mat_path).RELEASE
    annolist = np.atleast_1d(s.annolist)
    pool = util.BoundedPool(None, 120)

    for anno, is_train, rect_ids in zip(annolist, util.progressbar(s.img_train), s.single_person):
        if not is_train:
            continue

        image_path = os.path.join(paths.DATA_ROOT, 'mpii', 'images', anno.image.name)
        annorect = np.atleast_1d(anno.annorect)
        rect_ids = np.atleast_1d(rect_ids) - 1

        for rect_id in rect_ids:
            rect = annorect[rect_id]
            if 'annopoints' not in rect or len(rect.annopoints) == 0:
                continue

            coords = np.full(
                shape=[joint_info_full.n_joints, 2], fill_value=np.nan, dtype=np.float32)
            for joint in np.atleast_1d(rect.annopoints.point):
                coords[joint.id] = [joint.x, joint.y]

            coords = coords[selected_joints]
            rough_person_center = np.float32([rect.objpos.x, rect.objpos.y])
            rough_person_size = rect.scale * 200

            # Shift person center down like [Sun et al. 2018], who say this is common on MPII
            rough_person_center[1] += 0.075 * rough_person_size

            topleft = np.array(rough_person_center) - np.array(rough_person_size) / 2
            bbox = np.array([topleft[0], topleft[1], rough_person_size, rough_person_size])
            ex = Pose2DExample(image_path, coords, bbox=bbox)
            pool.apply_async(
                make_efficient_example, (ex, rect_id), callback=dataset.examples[TRAIN].append)

    print('Waiting for tasks...')
    pool.close()
    pool.join()
    print('Done...')
    dataset.examples[TRAIN].sort(key=lambda x: x.image_path)
    return dataset


def make_efficient_example(ex, rect_id):
    """Make example by storing the image in a cropped and resized version for efficient loading"""

    # Determine which area we will need
    # For rotation, usual padding around box, scale (shrink) augmentation and shifting
    padding_factor = 1 / 0.85
    scale_up_factor = 1 / 0.85
    scale_down_factor = 1 / 0.85
    shift_factor = 1.1
    max_rotate = np.pi / 6
    rot_factor = np.sin(max_rotate) + np.cos(max_rotate)
    base_dst_side = 256

    scale_factor = min(base_dst_side / ex.bbox[3] * scale_up_factor, 1)
    hopeful_factor = 0.9
    expansion_factor = (
            rot_factor * padding_factor * shift_factor * scale_down_factor * hopeful_factor)

    expanded_bbox = boxlib.expand(boxlib.expand_to_square(ex.bbox), expansion_factor)
    imsize = improc.image_extents(ex.image_path)
    full_box = np.array([0, 0, imsize[0], imsize[1]])
    expanded_bbox = boxlib.intersect(expanded_bbox, full_box)

    old_camera = cameralib.Camera.create2D()
    new_camera = old_camera.copy()
    new_camera.intrinsic_matrix[:2, 2] -= expanded_bbox[:2]
    new_camera.scale_output(scale_factor)

    dst_shape = improc.rounded_int_tuple(scale_factor * expanded_bbox[[3, 2]])
    new_im_path = ex.image_path.replace('mpii', f'mpii_downscaled')
    without_ext, ext = os.path.splitext(new_im_path)
    new_im_path = f'{without_ext}_{rect_id:02d}{ext}'

    if not (util.is_file_newer(new_im_path, "2019-11-12T17:54:06") and
            improc.is_image_readable(new_im_path)):
        im = improc.imread_jpeg_fast(ex.image_path)
        new_im = cameralib.reproject_image(im, old_camera, new_camera, dst_shape)
        util.ensure_path_exists(new_im_path)
        imageio.imwrite(new_im_path, new_im)

    new_bbox_topleft = cameralib.reproject_image_points(ex.bbox[:2], old_camera, new_camera)
    new_bbox = np.concatenate([new_bbox_topleft, ex.bbox[2:] * scale_factor])
    new_coords = cameralib.reproject_image_points(ex.coords, old_camera, new_camera)
    ex = Pose2DExample(os.path.relpath(new_im_path, paths.DATA_ROOT), new_coords, bbox=new_bbox)
    return ex


def current_dataset():
    from init import FLAGS
    return get_dataset(FLAGS.dataset)


@functools.lru_cache()
def get_dataset(dataset_name, *args, **kwargs):
    from init import FLAGS
    logging.debug(f'Making dataset {dataset_name}...')

    def string_to_intlist(string):
        return tuple(int(s) for s in string.split(','))

    kwargs = {**kwargs}
    for subj_key in ['train_subjects', 'valid_subjects', 'test_subjects']:
        if getattr(FLAGS, subj_key):
            kwargs[subj_key] = string_to_intlist(getattr(FLAGS, subj_key))

    return globals()[f'make_{dataset_name}'](*args, **kwargs)

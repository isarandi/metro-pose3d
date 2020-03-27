import copy
import functools
import itertools
import os
import os.path
import xml.etree.ElementTree

import imageio
import numpy as np
import spacepy.pycdf
import transforms3d

import boxlib
import cameralib
import data.datasets as ps3d
import improc
import paths
import util


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/h36m.pkl', min_time="2019-11-14T23:33:29")
def make_h36m(
        train_subjects=(1, 5, 6, 7, 8), valid_subjects=(), test_subjects=(9, 11),
        correct_S9=True, partial_visibility=False):
    joint_names = (
        'rhip,rkne,rank,lhip,lkne,lank,tors,neck,head,htop,'
        'lsho,lelb,lwri,rsho,relb,rwri,pelv'.split(','))

    j = ps3d.JointInfo.make_id_map(joint_names)
    edges = [
        (j.htop, j.head), (j.head, j.neck), (j.lsho, j.neck), (j.lelb, j.lsho), (j.lwri, j.lelb),
        (j.rsho, j.neck), (j.relb, j.rsho), (j.rwri, j.relb), (j.neck, j.tors), (j.tors, j.pelv),
        (j.lhip, j.pelv), (j.lkne, j.lhip), (j.lank, j.lkne), (j.rhip, j.pelv), (j.rkne, j.rhip),
        (j.rank, j.rkne)]
    joint_info = ps3d.JointInfo(j, edges)

    if not util.all_disjoint(train_subjects, valid_subjects, test_subjects):
        raise Exception('Set of train, val and test subject must be disjoint.')

    # use last subject of the non-test subjects for validation
    train_examples = []
    test_examples = []
    valid_examples = []
    pool = util.BoundedPool(None, 120)

    if partial_visibility:
        dir_suffix = '_partial'
        further_expansion_factor = 1.8
    else:
        dir_suffix = '' if correct_S9 else 'incorrect_S9'
        further_expansion_factor = 1

    for i_subject in [*test_subjects, *train_subjects, *valid_subjects]:
        if i_subject in train_subjects:
            examples_container = train_examples
        elif i_subject in valid_subjects:
            examples_container = valid_examples
        else:
            examples_container = test_examples

        frame_step = 5 if i_subject in train_subjects else 64

        for activity_name, camera_id in itertools.product(get_activity_names(i_subject), range(4)):
            print(f'Processing S{i_subject} {activity_name} {camera_id}')
            # Corrupt data in original release:
            if i_subject == 11 and activity_name == 'Directions' and camera_id == 0:
                continue

            data, camera = get_examples(
                i_subject, activity_name, camera_id, frame_step=frame_step, correct_S9=correct_S9)
            prev_coords = None
            for image_relpath, world_coords, bbox in data:
                # Using very similar examples is wasteful when training. Therefore:
                # skip frame if all keypoints are within a distance compared to last stored frame.
                # This is not done when testing, as it would change the results.
                if (i_subject in train_subjects and prev_coords is not None and
                        np.all(np.linalg.norm(world_coords - prev_coords, axis=1) < 100)):
                    continue
                prev_coords = world_coords
                activity_name = activity_name.split(' ')[0]
                ex = ps3d.Pose3DExample(
                    image_relpath, world_coords, bbox, camera, activity_name=activity_name)
                pool.apply_async(
                    make_efficient_example, (ex, further_expansion_factor, 1, dir_suffix),
                    callback=examples_container.append)

    print('Waiting for tasks...')
    pool.close()
    pool.join()
    print('Done...')
    train_examples.sort(key=lambda x: x.image_path)
    valid_examples.sort(key=lambda x: x.image_path)
    test_examples.sort(key=lambda x: x.image_path)
    return ps3d.Pose3DDataset(joint_info, train_examples, valid_examples, test_examples)


def make_efficient_example(ex, further_expansion_factor=1, further_scale_up=1, dir_suffix=''):
    """Make example by storing the image in a cropped and resized version for efficient loading"""

    # Determine which area we will need from the image
    # This is a bit larger than the tight crop because of the geometric augmentations
    max_rotate = np.pi / 6
    padding_factor = 1 / 0.85
    scale_up_factor = 1 / 0.85 * further_scale_up
    scale_down_factor = 1 / 0.85
    shift_factor = 1.1
    base_dst_side = 256

    box_center = boxlib.center(ex.bbox)
    s, c = np.sin(max_rotate), np.cos(max_rotate)
    w, h = ex.bbox[2:]
    rot_bbox_side = max(c * w + s * h, c * h + s * w)
    rot_bbox = boxlib.box_around(box_center, rot_bbox_side)

    scale_factor = min(base_dst_side / np.max(ex.bbox[2:]) * scale_up_factor, 1)
    expansion_factor = (
            padding_factor * shift_factor * scale_down_factor * further_expansion_factor)
    expanded_bbox = boxlib.expand(rot_bbox, expansion_factor)
    expanded_bbox = boxlib.intersect(expanded_bbox, np.array([0, 0, 1000, 1000]))

    new_camera = copy.deepcopy(ex.camera)
    new_camera.intrinsic_matrix[:2, 2] -= expanded_bbox[:2]
    new_camera.scale_output(scale_factor)
    new_camera.undistort()

    new_im_relpath = ex.image_path.replace('h36m', f'h36m_downscaled{dir_suffix}')
    new_im_path = f'{paths.DATA_ROOT}/{new_im_relpath}'
    if not (util.is_file_newer(new_im_path, "2019-11-14T23:33:14") and
            improc.is_image_readable(new_im_path)):
        im = improc.imread_jpeg(ex.image_path)
        dst_shape = improc.rounded_int_tuple(scale_factor * expanded_bbox[[3, 2]])
        new_im = cameralib.reproject_image(im, ex.camera, new_camera, dst_shape)
        util.ensure_path_exists(new_im_path)
        imageio.imwrite(new_im_path, new_im)

    new_bbox_topleft = cameralib.reproject_image_points(ex.bbox[:2], ex.camera, new_camera)
    new_bbox = np.concatenate([new_bbox_topleft, ex.bbox[2:] * scale_factor])
    ex = ps3d.Pose3DExample(
        new_im_relpath, ex.world_coords, new_bbox, new_camera, activity_name=ex.activity_name)
    return ex


def correct_boxes(bboxes, path, world_coords, camera):
    """Two activties for subject S9 have erroneous bounding boxes, they are horizontally shifted.
    This function corrects them. Use --dataset=h36m-incorrect-S9 to use the erroneous annotation."""

    def correct_image_coords(bad_imcoords):
        root_depths = camera.world_to_camera(world_coords[:, -1])[:, 2:]
        bad_worldcoords = camera.image_to_world(bad_imcoords, camera_depth=root_depths)
        good_worldcoords = bad_worldcoords + np.array([-200, 0, 0])
        good_imcoords = camera.world_to_image(good_worldcoords)
        return good_imcoords

    if 'S9' in path and ('SittingDown 1' in path or 'Waiting 1' in path or 'Greeting.' in path):
        toplefts = correct_image_coords(bboxes[:, :2])
        bottomrights = correct_image_coords(bboxes[:, :2] + bboxes[:, 2:])
        return np.concatenate([toplefts, bottomrights - toplefts], axis=-1)

    return bboxes


def correct_world_coords(coords, path):
    """Two activties for subject S9 have erroneous coords, they are horizontally shifted.
    This corrects them. Use --dataset=h36m-incorrect-S9 to use the erroneous annotation."""
    if 'S9' in path and ('SittingDown 1' in path or 'Waiting 1' in path or 'Greeting.' in path):
        coords = coords.copy()
        coords[:, :, 0] -= 200
    return coords


def load_cdf(path):
    try:
        spacepy.pycdf.CDF(path)
    except:
        print(path)
        raise


def get_examples(
        i_subject, activity_name, i_camera, frame_step=5, correct_S9=True):
    camera_names = ['54138969', '55011271', '58860488', '60457274']
    camera_name = camera_names[i_camera]
    h36m_root = f'{paths.DATA_ROOT}/h36m/'
    camera = get_cameras(f'{h36m_root}/Release-v1.2/metadata.xml')[i_camera][i_subject - 1]

    def load_coords(path):
        with spacepy.pycdf.CDF(path) as cdf_file:
            coords_raw_all = np.array(cdf_file['Pose'], np.float32)[0]
        coords_raw = coords_raw_all[::frame_step]
        i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
        coords_new_shape = [coords_raw.shape[0], -1, 3]
        return coords_raw_all.shape[0], coords_raw.reshape(coords_new_shape)[:, i_relevant_joints]

    pose_folder = f'{h36m_root}/S{i_subject}/MyPoseFeatures'
    coord_path = f'{pose_folder}/D3_Positions/{activity_name}.cdf'
    n_total_frames, world_coords = load_coords(coord_path)

    if correct_S9:
        world_coords = correct_world_coords(world_coords, coord_path)

    image_relfolder = f'h36m/S{i_subject}/Images/{activity_name}.{camera_name}'
    image_relpaths = [f'{image_relfolder}/frame_{i_frame:06d}.jpg'
                   for i_frame in range(0, n_total_frames, frame_step)]

    bbox_path = f'{h36m_root}/S{i_subject}/BBoxes/{activity_name}.{camera_name}.npy'
    bboxes = np.load(bbox_path)[::frame_step]
    if correct_S9:
        bboxes = correct_boxes(bboxes, bbox_path, world_coords, camera)

    return zip(image_relpaths, world_coords, bboxes), camera


@functools.lru_cache()
def get_cameras(metadata_path):
    root = xml.etree.ElementTree.parse(metadata_path).getroot()
    cam_params_text = root.findall('w0')[0].text
    numbers = np.array([float(x) for x in cam_params_text[1:-1].split(' ')])
    extrinsic = numbers[:264].reshape(4, 11, 6)
    intrinsic = numbers[264:].reshape(4, 9)

    cameras = [[make_h36m_camera(extrinsic[i_camera, i_subject], intrinsic[i_camera])
                for i_subject in range(11)]
               for i_camera in range(4)]
    return cameras


def make_h36m_camera(extrinsic_params, intrinsic_params):
    x_angle, y_angle, z_angle = extrinsic_params[0:3]
    R = transforms3d.euler.euler2mat(x_angle, y_angle, z_angle, 'rxyz')
    t = extrinsic_params[3:6]
    f, c, k, p = np.split(intrinsic_params, (2, 4, 7))
    distortion_coeffs = np.array([k[0], k[1], p[0], p[1], k[2]], np.float32)
    intrinsic_matrix = np.array([
        [f[0], 0, c[0]],
        [0, f[1], c[1]],
        [0, 0, 1]], np.float32)
    return cameralib.Camera(t, R, intrinsic_matrix, distortion_coeffs)


def get_activity_names(i_subject):
    h36m_root = f'{paths.DATA_ROOT}/h36m/'
    subject_images_root = f'{h36m_root}/S{i_subject}/Images/'
    subdirs = [elem for elem in os.listdir(subject_images_root)
               if os.path.isdir(f'{subject_images_root}/{elem}')]

    activity_names = set(elem.split('.')[0] for elem in subdirs if '_' not in elem)
    return sorted(activity_names)

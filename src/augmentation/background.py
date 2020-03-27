import functools

import numpy as np

import cameralib
import improc
import paths
import util


@functools.lru_cache()
def get_inria_holiday_background_paths():
    inria_holidays_root = f'{paths.DATA_ROOT}/inria_holidays'
    filenames = util.read_file(f'{inria_holidays_root}/non_person_images.txt').splitlines()
    return sorted(f'{inria_holidays_root}/jpg_small/{filename}' for filename in filenames)


def augment_background(im, fgmask, rng):
    path = util.choice(get_inria_holiday_background_paths(), rng)
    background_im = improc.imread_jpeg(path)

    cam = cameralib.Camera.create2D(background_im.shape)
    cam_new = cam.copy()

    zoom_aug_factor = rng.uniform(1.2, 1.5)
    cam_new.zoom(zoom_aug_factor * np.max(im.shape[:2] / np.asarray(background_im.shape[:2])))
    cam_new.center_principal_point(im.shape)
    cam_new.shift_image(util.random_uniform_disc(rng) * im.shape[0] * 0.1)
    warped_background_im = cameralib.reproject_image(background_im, cam, cam_new, im.shape)
    return improc.blend_image(warped_background_im, im, fgmask)

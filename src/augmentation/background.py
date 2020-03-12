import functools

import geom
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
    background_im = improc.imread_jpeg_fast(path)

    tr = geom.SimTransform()
    imside = im.shape[0]
    tr = (tr.center_fill(background_im.shape[:2], im.shape[:2], factor=rng.uniform(1.2, 1.5)).
          translate(rng.uniform(-imside * 0.1, imside * 0.1, size=2)))
    warped_background_im = tr.transform_image(background_im, dst_shape=im.shape[:2])
    return improc.blend_image(warped_background_im, im, fgmask)

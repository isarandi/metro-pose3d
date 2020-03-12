import cv2
import numpy as np
import skimage.transform as sktr


def transform_points_affine(points_as_rows, matrix):
    """Transforms D-dimensional points by a (D-1)xD or (D-1)x(D-1) matrix.
    
    Args:
        points_as_rows: input points to be transformed, shaped NxD
        matrix: forward transformation matrix from input to output, shaped (D-1)xD or (D-1)x(D-1)

    Returns:
        The resulting points as rows (shape Nx2).
    """
    point_dim = points_as_rows.shape[1]
    return cv2.transform(points_as_rows[np.newaxis], matrix[:point_dim])[0]


class SimTransform:
    """2D image transform. Applies operations (rot, scale, ...) by
     left-multiplying with the new operation's matrix representation."""

    def __init__(self, matrix=None, copy=True):
        if isinstance(matrix, SimTransform):
            mat = matrix.matrix
        elif matrix is not None:
            mat = matrix
        else:
            mat = np.identity(3)

        self.matrix = mat if not copy else mat.copy()

    def scale(self, scale):
        return SimTransform(sktr.SimilarityTransform(scale=scale).params @ self.matrix)

    def translate(self, translation):
        return SimTransform(sktr.SimilarityTransform(translation=translation).params @ self.matrix)

    def rotate(self, angle_radian):
        return SimTransform(sktr.SimilarityTransform(rotation=angle_radian).params @ self.matrix)

    def as_skimage_transform(self):
        return sktr.SimilarityTransform(matrix=self.matrix)

    def horizontal_flip(self):
        matrix = self.matrix.copy()
        matrix[0:2, 0] *= -1
        return SimTransform(matrix)

    def get_scale_factor(self):
        # Assuming similarity transformation
        return np.linalg.norm(self.matrix[:, 0])

    def is_flipped(self):
        return np.linalg.det(self.matrix[:2, :2]) < 0

    def inverse(self):
        return SimTransform(np.linalg.inv(self.matrix))

    def copy(self):
        return self.__class__(self.matrix, copy=True)

    def center_fit(self, src_shape, dst_shape):
        return (self.translate(-np.asarray(src_shape[::-1]) / 2).
                scale(np.min(np.asarray(dst_shape) / np.asarray(src_shape))).
                translate(np.asarray(dst_shape[::-1]) / 2))

    def center_fill(self, src_shape, dst_shape, factor=1):
        return (self.translate(-np.asarray(src_shape[::-1]) / 2).
                scale(factor * np.max(np.asarray(dst_shape) / np.asarray(src_shape))).
                translate(np.asarray(dst_shape[::-1]) / 2))

    def __matmul__(self, other):
        if isinstance(other, SimTransform):
            return self.__class__(self.matrix @ other.matrix)
        else:
            return self.__class__(self.matrix @ other)

    def __rmatmul__(self, other):
        if isinstance(other, SimTransform):
            return self.__class__(other.matrix @ self.matrix)
        else:
            return self.__class__(other @ self.matrix)

    def transform_image(self, im, border_value=0, dst_shape=None):
        if dst_shape is None:
            dst_shape = im.shape[:2]

        # return warp_affine(im, self.matrix, (dst_shape[1], dst_shape[0]),
        # borderValue=border_value)
        scaling_factor = np.linalg.norm(self.matrix[:2, 0])
        interp = cv2.INTER_LINEAR if scaling_factor > 1 else cv2.INTER_AREA
        return cv2.warpAffine(
            im, self.matrix[:2], (dst_shape[1], dst_shape[0]), flags=interp,
            borderValue=border_value)

    def transform_points(self, points_as_rows):
        return transform_points_affine(points_as_rows, self.matrix)

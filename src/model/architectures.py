import tensorflow as tf
import tensorflow.contrib.slim as slim

import model.mobilenet_v3
import model.resnet_v2
import tfu
from init import FLAGS


def resnet_arg_scope():
    batch_norm_params = dict(
        decay=0.997, epsilon=1e-5, scale=True, is_training=tfu.is_training(), fused=True,
        data_format=tfu.data_format())

    with slim.arg_scope(
            [slim.conv2d, slim.conv3d],
            weights_regularizer=slim.l2_regularizer(1e-4),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@tfu.in_variable_scope('Resnet', mixed_precision=True)
def resnet(inp, n_out, stride=16, centered_stride=False, resnet_name='resnet_v2_50'):
    # if resnet_name == 'mobilenet':
    #     return mobilenet(inp, n_out, stride)
    with slim.arg_scope(resnet_arg_scope()):
        inp = inp * 2 - 1
        x = tf.cast(inp, tfu.get_dtype())
        resnet_fn = getattr(model.resnet_v2, resnet_name)
        x, end_points = resnet_fn(
            x, num_classes=n_out, is_training=tfu.is_training(), global_pool=False,
            output_stride=stride, centered_stride=centered_stride)
        x = tf.cast(x, tf.float32)
        return x


# @tfu.in_variable_scope('Mobilenet', mixed_precision=True)
# def mobilenet(inp, n_out, stride=16):
#     with slim.arg_scope(model.mobilenet_v3.training_scope(is_training=tfu.is_training())):
#         inp = inp * 2 - 1
#         x = tf.cast(inp, tfu.get_dtype())
#         print(tfu.static_shape(x))
#         x, end_points = model.mobilenet_v3.mobilenet(
#             x, conv_defs=model.mobilenet_v3.V3_LARGE, finegrain_classification_mode=False,
#             base_only=True, output_stride=stride, use_explicit_padding=True)
#         print(tfu.static_shape(x))
#         x = slim.conv2d(
#             x, n_out, 1, activation_fn=None, normalizer_fn=None,
#             biases_initializer=tf.compat.v1.zeros_initializer(), scope='logits')
#         x = tf.cast(x, tf.float32)
#         return x

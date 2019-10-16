# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import random
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")


def decode_jpeg(image_buffer, scope=None):
    with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def distort_color(image, thread_id=0, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            image = tf.image.random_hue(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            image = tf.image.random_hue(image, max_delta=0.1)

        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def random_pixel_value_scale(image, minval=0.8, maxval=1.2, seed=None):
    """
    随机改变图像的每一个像素值（只进行轻微改变）
    输入的图像张量是tf.float32类型的，且每一个像素值都被归一化到0-1的范围
    """
    with tf.name_scope('RandomPixelValueScale', values=[image]):
        color_coef = tf.random_uniform(tf.shape(image),
            minval=minval,
            maxval=maxval,
            dtype=tf.float32,
            seed=seed)
        image = tf.multiply(image,color_coef)
        image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def random_crop(image):
    """
    图像随机裁剪，py_func函数np
    """
    h, w, c = image.shape
    random_h_init = random.randint(0, int(h*0.18))
    random_w_init = random.randint(0, int(w*0.18))
    random_h_end = random.randint(0, int(h*0.18))
    random_w_end = random.randint(0, int(w*0.18))

    image = image[random_h_init:h-random_h_end, random_w_init:w-random_w_end, :]
    return image


def image_padding(image):
    """
    将图像padding到固定长宽比
    """
    aspect_ratio = 1.0
    h, w = image.shape[0:2]
    if h > aspect_ratio * w:
        pad_left = int((h - w) / 2)
        pad_right = int(h - pad_left - w)
        image = np.pad(image, ((0,0), (pad_left, pad_right), (0,0)), 'constant', constant_values=(0,0))
    elif w > aspect_ratio * h:
        pad_up = int((w - h) / 2)
        pad_down = int(w - pad_up - h)
        image = np.pad(image, ((pad_up,pad_down), (0,0), (0,0)), 'constant', constant_values=(0,0))
    else:
        pass
    return image


def distort_image(image, height, width, bbox, thread_id=0, scope=None):
    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        distorted_image = image
        method = random.randint(0,3)

        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)

        # 随机改变部分像素值
        # distorted_image = random_pixel_value_scale(distorted_image)

        # 图像随机裁剪
        distorted_image = tf.py_func(random_crop, [distorted_image], tf.float32)
        distorted_image.set_shape([None, None, 3])

        # 图像padding到固定长宽比
        distorted_image = tf.py_func(image_padding, [distorted_image], tf.float32)
        distorted_image.set_shape([None, None, 3])

        distorted_image = tf.expand_dims(distorted_image, 0)

        # 图像resize
        distorted_image = tf.image.resize_images(distorted_image, [height, width], method)

        tf.summary.image('preprocessed_image', distorted_image)
        image = tf.squeeze(image, [0])
        return distorted_image


def eval_image(image, height, width, scope=None):
    """Prepare one image for evaluation.

    Args:
      image: 3-D float Tensor
      height: integer
      width: integer
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        image = tf.image.central_crop(image, central_fraction=0.9)

        # 图像padding到固定长宽比
        image = tf.py_func(image_padding, [image], tf.float32)
        image.set_shape([None, None, 3])

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        return image


def image_preprocessing(image, bbox, train, thread_id=0):
    if bbox is None:
        raise ValueError('Please supply a bounding box.')

    if train:
        image = decode_jpeg(image)
    height = 128
    width = 128

    if train:
        image = distort_image(image, height, width, bbox, thread_id)
    else:
        image = eval_image(image, height, width)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
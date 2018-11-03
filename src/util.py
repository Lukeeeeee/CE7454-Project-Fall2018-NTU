import tensorflow as tf
from utils.config import Config
from src.model import ICNet_BN
from utils.config import Config
from utils.image_reader import ImageReader, prepare_label
from PIL import Image
import os
import numpy as np


def save_pred_to_image(res, shape, save_path, save_name):
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    res = np.array(np.reshape(res, shape), dtype=np.uint8) * 255
    img = Image.fromarray(res.astype(np.uint8), mode='L')
    img.save(os.path.join(save_path, save_name))


def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes - 1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices


def create_bce_loss(output, label, num_classes, ignore_label):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1, ])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    gt_one_hot = tf.one_hot(gt, num_classes)
    pred = tf.gather(raw_pred, indices)
    BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=gt_one_hot))

    inse = tf.reduce_sum(pred * tf.cast(gt_one_hot, tf.float32))
    l = tf.reduce_sum(pred)
    r = tf.reduce_sum(tf.cast(gt_one_hot, tf.float32))
    dice = tf.math.log((2. * inse + 1e-5) / (l + r + 1e-5))

    loss = BCE - dice
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss


global_logits = []
global_label = []


def create_loss(output, label, num_classes, ignore_label):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1, ])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
    reduced_loss = tf.reduce_mean(loss)

    global_label.append(gt)
    global_logits.append(pred)

    return reduced_loss


def create_losses(net, label, cfg):
    # Get output from different branches
    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_cls']
    if cfg.param['loss_type'] == 'cross_entropy':

        loss_sub4 = create_loss(sub4_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
        loss_sub24 = create_loss(sub24_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
        loss_sub124 = create_loss(sub124_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
    elif cfg.para['loss_type'] == 'bce':
        loss_sub4 = create_bce_loss(sub4_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
        loss_sub24 = create_bce_loss(sub24_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
        loss_sub124 = create_bce_loss(sub124_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
    else:
        raise ValueError('Loss type not exisited, use "cross_entropy" or "bce"')

    l2_losses = [cfg.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]

    # Calculate weighted loss of three branches, you can tune LAMBDA values to get better results.
    reduced_loss = cfg.LAMBDA1 * loss_sub4 + cfg.LAMBDA2 * loss_sub24 + cfg.LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

    return loss_sub4, loss_sub24, loss_sub124, reduced_loss

"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""
import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)

import argparse
import time
import math
import tensorflow as tf
from utils.config import Config

from src.model import ICNet_BN
from utils.config import Config
from utils.image_reader import ImageReader, prepare_label


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced ICNet")

    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--dataset", required=True,
                        help="Which dataset to trained with",
                        choices=['cityscapes', 'ade20k', 'others'])
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    return parser.parse_args()


def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes - 1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices


def dice_coef_theoretical(y_pred, y_true):
    """Define the dice coefficient
        Args:
        y_pred: Prediction
        y_true: Ground truth Label
        Returns:
        Dice coefficient
        """

    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)

    # y_pred_f = tf.nn.sigmoid(y_pred)
    # y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)
    # y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)

    y_pred_f = tf.nn.softmax(y_pred)
    y_pred_f = tf.argmax(y_pred_f, axis=1)
    y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection) / (union + 0.00001)

    if (tf.reduce_sum(y_pred) == 0) and (tf.reduce_sum(y_true) == 0):
        dice = 1

    return dice


def create_bce_loss(output, label, num_classes, ignore_label):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1, ])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    gt_one_hot = tf.one_hot(gt, num_classes)
    pred = tf.gather(raw_pred, indices)
    BCE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=gt_one_hot))

    # inse = tf.reduce_sum(pred * tf.cast(gt_one_hot,tf.float32))
    # l = tf.reduce_sum(pred)
    # r = tf.reduce_sum(tf.cast(gt_one_hot,tf.float32))
    # dice = tf.math.log((2. * inse + 1e-5) / (l + r + 1e-5))
    dice = dice_coef_theoretical(pred, gt)
    # tf.Print(dice)
    loss = BCE - tf.math.log(dice)
    reduced_loss = loss

    return reduced_loss


def create_loss(output, label, num_classes, ignore_label):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1, ])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss


def create_losses(net, label, cfg):
    # Get output from different branches
    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_cls']

    loss_sub4 = create_loss(sub4_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
    loss_sub24 = create_loss(sub24_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])
    loss_sub124 = create_loss(sub124_out, label, cfg.param['num_classes'], cfg.param['ignore_label'])

    l2_losses = [cfg.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]

    # Calculate weighted loss of three branches, you can tune LAMBDA values to get better results.
    reduced_loss = cfg.LAMBDA1 * loss_sub4 + cfg.LAMBDA2 * loss_sub24 + cfg.LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

    return loss_sub4, loss_sub24, loss_sub124, reduced_loss


class TrainConfig(Config):
    def __init__(self, dataset, is_training, filter_scale=1, random_scale=None, random_mirror=None, log_path_end='',
                 model_weight=None):
        Config.__init__(self, dataset, is_training, filter_scale, random_scale, random_mirror,
                        log_path_end=log_path_end)

        # Set pre-trained weights here (You can download weight using `python script/download_weights.py`)
        # Note that you need to use "bnnomerge" version.
        # model_weight = '../model/cityscapes/icnet_cityscapes_train_30k_bnnomerge.npy'
        if not model_weight:
            self.model_weight = '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-09_21-00-37_v2_DEFAULT_CONFIG_EPOCH_10/model.ckpt-9'
        else:
            self.model_weight = model_weight

    # Set hyperparameters here, you can get much more setting in Config Class, see 'utils/config.py' for details.
    LAMBDA1 = 0.16
    LAMBDA2 = 0.4
    LAMBDA3 = 1.0
    # previously 8
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-4


def main(lr=None, log_path_end='', bs=None, train_epoch=None, lambda_list=None, random_mirror=False, random_scale=False,
         model_weight=None):
    """Create the model and start the training."""
    tf.reset_default_graph()
    args = get_arguments()

    """
    Get configurations here. We pass some arguments from command line to init configurations, for training hyperparameters, 
    you can set them in TrainConfig Class.

    Note: we set filter scale to 1 for pruned model, 2 for non-pruned model. The filters numbers of non-pruned
          model is two times larger than prunde model, e.g., [h, w, 64] <-> [h, w, 32].
    """
    cfg = TrainConfig(dataset=args.dataset,
                      is_training=True,
                      random_scale=random_scale,
                      random_mirror=random_mirror,
                      filter_scale=args.filter_scale,
                      log_path_end=log_path_end,
                      model_weight=model_weight)
    if lr:
        cfg.LEARNING_RATE = lr
    if bs:
        cfg.BATCH_SIZE = bs

    if lambda_list:
        cfg.LAMBDA1 = lambda_list[0]
        cfg.LAMBDA2 = lambda_list[1]
        cfg.LAMBDA3 = lambda_list[2]
    if train_epoch is not None:
        cfg.TRAINING_EPOCHS = train_epoch

    cfg.display()

    # Setup training network and training samples
    train_reader = ImageReader(cfg=cfg, mode='train')
    train_net = ICNet_BN(image_reader=train_reader,
                         cfg=cfg, mode='train')

    loss_sub4, loss_sub24, loss_sub124, reduced_loss = create_losses(train_net, train_net.labels, cfg)

    # Setup validation network and validation samples

    with tf.variable_scope('', reuse=True):
        val_reader = ImageReader(cfg, mode='eval')
        val_net = ICNet_BN(image_reader=val_reader,
                           cfg=cfg, mode='train')

        val_loss_sub4, val_loss_sub24, val_loss_sub124, val_reduced_loss = create_losses(val_net, val_net.labels, cfg)

    # Using Poly learning rate policy 
    base_lr = tf.constant(cfg.LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / cfg.TRAINING_EPOCHS), cfg.POWER))
    # learning_rate = base_lr

    # Set restore variable
    restore_var = tf.global_variables()
    restore_var = [v for v in tf.global_variables() if 'conv6_cls' not in v.name]
    all_trainable = [v for v in tf.trainable_variables() if
                     ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]

    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, cfg.MOMENTUM)
        grads = tf.gradients(reduced_loss, all_trainable)
        train_op = opt_conv.apply_gradients(zip(grads, all_trainable))

    # Create session & restore weights (Here we only need to use train_net to create session since we reuse it)
    train_net.create_session()
    train_net.restore(cfg.model_weight, restore_var)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

    # Iterate over training steps.
    train_info = []
    iter_max = math.ceil(cfg.param['total_train_sample'] / cfg.BATCH_SIZE)
    try:
        for epoch in range(cfg.TRAINING_EPOCHS):
            feed_dict = {step_ph: epoch}
            for iter in range(iter_max):
                start_time = time.time()
                loss_value, loss1, loss2, loss3, val_loss_value, _ = train_net.sess.run(
                    [reduced_loss, loss_sub4, loss_sub24, loss_sub124, val_reduced_loss, train_op],
                    feed_dict=feed_dict)
                duration = time.time() - start_time
                log = {
                    'LOSS_VALUE': float(loss_value),
                    'LOSS_1': float(loss1),
                    'LOSS_2': float(loss2),
                    'LOSS_3': float(loss3),
                    'VALIDATION_LOSS_VALUE': float(val_loss_value),
                    'DURATION': float(duration),
                    'STEP': int(iter),
                    'EPOCH': int(epoch),
                }
                train_info.append(log)
                print(
                    'epoch {:d} step {:d} \t total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f}, val_loss: {:.3f} ({:.3f} sec/step)'. \
                        format(epoch, iter, loss_value, loss1, loss2, loss3, val_loss_value, duration))
            if (epoch + 1) % cfg.SAVE_PRED_EVERY == 0:
                train_net.save(saver, cfg.SNAPSHOT_DIR, epoch)

    except KeyboardInterrupt:
        Config.save_to_json(dict=train_info, path=cfg.SNAPSHOT_DIR, file_name='loss.json')
        print("loss.json was saved at %s" % cfg.SNAPSHOT_DIR)
    Config.save_to_json(dict=train_info, path=cfg.SNAPSHOT_DIR, file_name='loss.json')
    print("loss.json was saved at %s" % cfg.SNAPSHOT_DIR)
    sess = tf.get_default_session()
    if sess:
        sess._exit__(None, None, None)
    return cfg.SNAPSHOT_DIR


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    lr_list = [5e-4, 1e-4, 5e-3, 1e-3, 5e-3, 1e-2, 5e-2]
    bs_list = [64, 32]
    lambda_list = [
        [0.16, 0.4, 1.0],

        [0.16, 0.5, 1.0],
        [0.16, 0.3, 1.0],

        [0.20, 0.4, 1.0],
        [0.12, 0.4, 1.0],

        [0.16, 0.4, 1.2],
        [0.16, 0.4, 0.8],
    ]
    train_step = [5, 10, 20, 40, 80, 100, 200]

    # for lamd in lambda_list:
    #     main(lambda_list=lamd, log_path_end='DEFAULT_CONFIG_LOSS_LAMBDA_%f_%f_%f' % (lamd[0], lamd[1], lamd[2]))
    # main(lr=5e-3, log_path_end='v2_DEFAULT_CONFIG_LR_%f' % 5e-3)
    # for tr in train_step:
    #     main(train_epoch=tr, log_path_end='v2_DEFAULT_CONFIG_EPOCH_%d' % tr)

    # main(log_path_end='v2_best_hyper_parameter_epoch_20',
    #      random_scale=False,
    #      lambda_list=[0.16, 0.3, 1.0],
    #      lr=0.000500,
    #      train_epoch=20)

    main(log_path_end='v2_best_hyper_parameter_epoch_200_continue',
         random_scale=False,
         lambda_list=[0.16, 0.3, 1.0],
         model_weight='/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-13_01-11-43_v2_best_hyper_parameter_epoch_200/model.ckpt-154',
         lr=0.000500,
         train_epoch=45)

    # main(log_path_end='v2_restore_2018-11-09_21-00-37_random_mirror_10_extra_epoch',
    #      random_mirror=True,
    #      train_epoch=10)
    #
    # log_dir = main(log_path_end='v2_restore_2018-11-09_21-00-37_random_scale_5_extra_epoch',
    #                random_scale=True,
    #                train_epoch=5)
    #
    # main(log_path_end='v2_restore_2018-11-09_21-00-37_random_mirror_5_extra_epoch',
    #      random_mirror=True,
    #      model_weight=os.path.join(log_dir, 'model.ckpt-4'),
    #      train_epoch=5)

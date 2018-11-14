import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)
import argparse
import tensorflow as tf
from tqdm import trange
from utils.image_reader import _image_mirroring, _random_crop_and_pad_image_and_labels, _image_scaling
from utils.config import Config
from utils.image_reader import ImageReader
from src.model import ICNet, ICNet_BN
import os
from log import LOG_PATH
import numpy as np
from src.util import save_pred_to_image
import argparse
import tensorflow as tf
import numpy as np
import cv2
import time
from data import DATA_PATH
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange
from utils.config import Config
import glob
from data.tnet_offline_validation_set_res import TNET_LOG_PATH
import torch

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}
IMG_MEAN = np.array((177.682, 175.84, 174.21), dtype=np.float32)


def _extract_mean_revert(img, img_mean, swap_channel=False):
    # swap channel and extract mean
    img += img_mean
    if swap_channel:
        img_b = img[:, :, 0]
        img_g = img[:, :, 1]
        img_r = img[:, :, 2]

        img_b = img_b[:, :, np.newaxis]
        img_r = img_r[:, :, np.newaxis]
        img_g = img_g[:, :, np.newaxis]

        img = np.concatenate((img_r, img_g, img_b), axis=2)

    return img


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced ICNet")

    parser.add_argument("--model", type=str, default='others',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'],
                        required=False)
    parser.add_argument("--dataset", type=str, default='others',
                        choices=['ade20k', 'cityscapes', 'others'],
                        required=False)
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])

    return parser.parse_args()


def main(model_log_dir, check_point):
    tf.reset_default_graph()

    args = get_arguments()

    cfg = Config(dataset=args.dataset,
                 is_training=False,
                 filter_scale=args.filter_scale,
                 eval_path_log=os.path.join(LOG_PATH, model_log_dir))
    cfg.model_paths['others'] = os.path.join(LOG_PATH, model_log_dir, 'model.ckpt-%d' % check_point)

    model = model_config[args.model]

    reader = ImageReader(cfg=cfg, mode='eval')
    net = model(image_reader=reader, cfg=cfg, mode='eval')

    # mIoU
    pred_flatten = tf.reshape(net.output, [-1, ])
    label_flatten = tf.reshape(net.labels, [-1, ])

    mask = tf.not_equal(label_flatten, cfg.param['ignore_label'])
    indices = tf.squeeze(tf.where(mask), 1)
    gt = tf.cast(tf.gather(label_flatten, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    tnet_result = np.load(file=os.path.join(TNET_LOG_PATH, 'valid.npy'))

    weight_list = [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]]
    ensemble_pred_list = []
    ensemble_input = tf.placeholder(dtype=pred.dtype, shape=[None])

    for weight in weight_list:
        ensemble_pred = tf.split(net.logits_up, 2, axis=len(net.logits_up.get_shape()) - 1)[1] * weight[0]
        ensemble_pred = tf.gather(tf.reshape(ensemble_pred, [-1, ]), indices)
        ensemble_pred = ensemble_pred + tf.cast(ensemble_input, tf.float32) * tf.constant(weight[1])
        ensemble_pred = tf.round(ensemble_pred)
        ensemble_pred_list.append(ensemble_pred)

    ensemble_mIoU_list = []

    ensemble_update_op_list = []

    assert cfg.dataset == 'others'
    mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    for ensemble_pred in ensemble_pred_list:
        ensemble_mIoU, ensemble_update_op = tf.metrics.mean_iou(predictions=ensemble_pred, labels=gt,
                                                                num_classes=cfg.param['num_classes'])
        ensemble_mIoU_list.append(ensemble_mIoU)
        ensemble_update_op_list.append(ensemble_update_op)

    net.create_session()
    net.restore(cfg.model_paths[args.model])
    duration = 0

    for i in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):
        start = time.time()
        feed_dict = {ensemble_input: tnet_result[i]}
        _ = net.sess.run(
            [update_op] + ensemble_update_op_list,
            feed_dict=feed_dict)
        end = time.time()
        duration += (end - start)

    final_mIou = net.sess.run(mIoU)
    ensemble_final_mIou_list = net.sess.run(ensemble_mIoU_list)

    print('total time:{} mean inference time:{} mIoU: {}'.format(duration,
                                                                 duration / cfg.param['eval_steps'],
                                                                 final_mIou))
    for weight, ensemble_iou in zip(weight_list, ensemble_final_mIou_list):
        print(weight, ensemble_iou)

    Config.save_to_json(dict={'FINAL_MIOU': float(final_mIou),
                              "EVAL_STEPS": cfg.param['eval_steps'],
                              "ENSEMBLE_WEIGHT": weight_list,
                              "ENSEMBLE_MIOU": [float(x) for x in ensemble_final_mIou_list]},
                        path=os.path.dirname(cfg.model_paths['others']),
                        file_name='eval.json',
                        mode='eval')
    sess = tf.get_default_session()
    if sess:
        sess._exit__(None, None, None)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    main(model_log_dir='2018-11-12_22-31-51_v2_best_hyper_parameter_epoch_20', check_point=19)

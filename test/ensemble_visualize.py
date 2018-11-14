import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)

from utils.image_reader import ImageReader
from src.model import ICNet, ICNet_BN
import os
from log import LOG_PATH
import argparse
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange
from utils.config import Config
from data.tnet_offline_validation_set_res import TNET_LOG_PATH

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


def main(model_log_dir, check_point, weight):
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
    pred = tf.gather(pred_flatten, indices)

    tnet_result = np.load(file=os.path.join(TNET_LOG_PATH, 'valid.npy'))

    assert cfg.dataset == 'others'

    duration = 0

    for i in trange(10, desc='evaluation', leave=True):
        start = time.time()
        icnet_res, input, labels = net.sess.run([pred, net.images, net.labels])

        end = time.time()
        duration += (end - start)

        tnet_result_i = tnet_result[i]

        ensemble = 0.4 * icnet_res + 0.6 * tnet_result_i
        ensemble[ensemble >= 0.5] = 1
        ensemble[ensemble < 0.5] = 0
        ensemble = np.array(np.reshape(ensemble, cfg.param['eval_size']), dtype=np.uint8) * 255
        ensemble_fig = Image.fromarray(ensemble.astype(np.uint8))

        input = np.squeeze(input)
        n_input = _extract_mean_revert(input, IMG_MEAN, swap_channel=True)
        n_input = n_input.astype(np.uint8)
        input_image = Image.fromarray(n_input, 'RGB')

        icnet = np.array(np.reshape(icnet_res, cfg.param['eval_size']), dtype=np.uint8) * 255
        icnet = Image.fromarray(icnet.astype(np.uint8))
        labels = np.squeeze(labels) * 255
        labels = Image.fromarray(labels.astype(np.uint8))
        fig, ax1 = plt.subplots(figsize=(10, 8))

        plot1 = plt.subplot(141)
        plot1.set_title("Input Image", fontsize=10)
        plt.imshow(input_image)
        plt.axis('off')

        plot2 = plt.subplot(142)
        plot2.set_title("Ground Truth Mask", fontsize=10)
        plt.imshow(labels, cmap='gray')
        plt.axis('off')

        plot3 = plt.subplot(143)
        plot3.set_title("Our Result", fontsize=10)
        plt.imshow(icnet, cmap='gray')
        plt.axis('off')

        plot4 = plt.subplot(144)
        plot4.set_title("Ensemble's Result", fontsize=10)
        plt.imshow(ensemble_fig, cmap='gray')
        plt.axis('off')

        plt.show()

        save_comparation_path = os.path.dirname(cfg.model_paths['others']) + '/eval_ensemble'
        if os.path.exists(save_comparation_path) is False:
            os.mkdir(save_comparation_path)
        plt.savefig(os.path.join(save_comparation_path, 'eval_%d_img.png' % i))

    sess = tf.get_default_session()
    if sess:
        sess._exit__(None, None, None)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    main(model_log_dir='2018-11-13_16-29-55_v2_best_hyper_parameter_epoch_200_continue',
         check_point=44,
         weight=[0.6, 0.4])

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

'''add'''
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
from test.TernausNet import Example
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

    # weight_list = [0.4, 0.6]
    # ensemble_input = tf.placeholder(dtype=pred.dtype, shape=pred.shape(), name='ensemble_input')
    # ensemble_pred = pred * weight_list[0] + ensemble_input * weight_list[1]

    if cfg.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'] + 1)
    elif cfg.dataset == 'cityscapes':
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    elif cfg.dataset == 'others':
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
        # ensemble_mIoU, ensemble_update_op = tf.metrics.mean_iou(predictions=ensemble_pred, labels=gt,
        #                                                         num_classes=cfg.param['num_classes'])

    net.create_session()
    net.restore(cfg.model_paths[args.model])

    # im1 = cv2.imread('/home/wei005/PycharmProjects/CE7454_Project_Fall2018_NTU/data/Kaggle/train/data/0cdf5b5d0ce1_05.jpg')
    # im2=cv2.imread('/home/wei005/PycharmProjects/CE7454_Project_Fall2018_NTU/data/Kaggle/train/mask/0cdf5b5d0ce1_05_mask.png',cv2.IMREAD_GRAYSCALE)
    # if im1.shape != cfg.INFER_SIZE:
    #     im1 = cv2.resize(im1, (cfg.INFER_SIZE[1], cfg.INFER_SIZE[0]))
    #
    # results1 = net.predict(im1)
    # overlap_results1 = 0.5 * im1 + 0.5 * results1[0]
    # vis_im1 = np.concatenate([im1 / 255.0, results1[0] / 255.0, overlap_results1 / 255.0], axis=1)

    # results1=results1[0][:,:,0]*255

    duration = 0
    model = Example.get_model()

    for i in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):
        start = time.time()
        _, res, input, labels, out, icnet_logit = net.sess.run(
            [update_op, pred, net.images, net.labels, net.output, net.logits_up])
        end = time.time()

        duration += (end - start)

        if i % 100 == 0:
            # save_pred_to_image(res=res,
            #                    shape=cfg.param['eval_size'],
            #                    save_path=os.path.dirname(cfg.model_paths['others']) + '/eval_img',
            #                    save_name='eval_%d_img.png' % i)

            input = np.squeeze(input)
            n_input = _extract_mean_revert(input, IMG_MEAN, swap_channel=True)
            n_input = n_input.astype(np.uint8)
            input_image = Image.fromarray(n_input, 'RGB')

            '''tnet -> tnet's predict either 0 1'''
            res2, tnet_mask = Example.ternauNet(n_input, model)
            tnet = Image.fromarray((tnet_mask * 255).astype(np.uint8))

            res2 = np.reshape(res2, [-1])
            icnet_logit = np.squeeze(icnet_logit)[:, :, 1]
            icnet_logit = np.reshape(icnet_logit, [-1])

            # TODO fix the ensemble problem: ensemble the output of softmax, not the argmax: fixed!
            ensemble = 0.4 * icnet_logit + 0.6 * res2
            ensemble[ensemble >= 0.5] = 1
            ensemble[ensemble < 0.5] = 0
            ensemble = np.array(np.reshape(ensemble, cfg.param['eval_size']), dtype=np.uint8) * 255
            ensemble_fig = Image.fromarray(ensemble.astype(np.uint8))
            # TODO save the ensemble result into picture: fixed!

            '''res-> network predict either 0 or 1 on each element '''
            icnet = np.array(np.reshape(res, cfg.param['eval_size']), dtype=np.uint8) * 255
            icnet = Image.fromarray(icnet.astype(np.uint8))
            labels = np.squeeze(labels) * 255
            labels = Image.fromarray(labels.astype(np.uint8))
            fig, ax1 = plt.subplots(figsize=(20, 12))

            plot1 = plt.subplot(141)
            plot1.set_title("Input Image", fontsize=4)
            plt.imshow(input_image)
            plt.axis('off')

            plot2 = plt.subplot(142)
            plot2.set_title("Ground Truth Mask", fontsize=4)
            plt.imshow(labels, cmap='gray')
            plt.axis('off')

            plot3 = plt.subplot(143)
            plot3.set_title("Our Result", fontsize=4)
            plt.imshow(icnet, cmap='gray')
            plt.axis('off')

            plot4 = plt.subplot(144)
            plot4.set_title("Ensemble's Result", fontsize=4)
            plt.imshow(ensemble_fig, cmap='gray')
            plt.axis('off')

            save_comparation_path = os.path.dirname(cfg.model_paths['others']) + '/eval_compare'
            if os.path.exists(save_comparation_path) is False:
                os.mkdir(save_comparation_path)
            plt.savefig(os.path.join(save_comparation_path, 'eval_%d_img.png' % i))
            plt.show()

    # TODO fix the mIou which take the ensemble as output: not done yet!
    final_mIou = net.sess.run(mIoU)
    # ensemble_final_mIou = net.sess.run(ensemble_mIoU)
    ensemble_final_mIou = -1.0

    print('total time:{} mean inference time:{} mIoU: {}, ensemble mIoU'.format(duration,
                                                                                duration / cfg.param['eval_steps'],
                                                                                final_mIou, ensemble_final_mIou))

    Config.save_to_json(dict={'FINAL_MIOU': float(final_mIou),
                              "EVAL_STEPS": cfg.param['eval_steps'],
                              "ENSEMBLE_MIOU": float(ensemble_final_mIou)},
                        path=os.path.dirname(cfg.model_paths['others']),
                        file_name='eval.json')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    main(model_log_dir='2018-11-08_10-52-15__v2_DEFAULT_CONFIG_LR_0.000100', check_point=19)

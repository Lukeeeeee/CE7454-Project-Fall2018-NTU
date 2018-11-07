import argparse

import tensorflow as tf
from tqdm import trange

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
import matplotlib.pyplot as plt



from tqdm import trange
from utils.config import Config

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}


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
    cfg.display()
    model = model_config[args.model]

    reader = ImageReader(cfg=cfg, mode='eval')
    net = model(cfg=cfg, mode='inference')

    # mIoU
    # pred_flatten = tf.reshape(net.output, [-1, ])
    # label_flatten = tf.reshape(net.labels, [-1, ])
    #
    # mask = tf.not_equal(label_flatten, cfg.param['ignore_label'])
    # indices = tf.squeeze(tf.where(mask), 1)
    # gt = tf.cast(tf.gather(label_flatten, indices), tf.int32)
    # pred = tf.gather(pred_flatten, indices)

    # if cfg.dataset == 'ade20k':
    #     pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
    #     mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'] + 1)
    # elif cfg.dataset == 'cityscapes':
    #     mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    # elif cfg.dataset == 'others':
    #     mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])

    net.create_session()
    net.restore(cfg.model_paths[args.model])

    im1 = cv2.imread('/home/wei005/PycharmProjects/CE7454_Project_Fall2018_NTU/data/Kaggle/train/data/0cdf5b5d0ce1_05.jpg')
    im2=cv2.imread('/home/wei005/PycharmProjects/CE7454_Project_Fall2018_NTU/data/Kaggle/train/mask/0cdf5b5d0ce1_05_mask.png',cv2.IMREAD_GRAYSCALE)
    if im1.shape != cfg.INFER_SIZE:
        im1 = cv2.resize(im1, (cfg.INFER_SIZE[1], cfg.INFER_SIZE[0]))

    results1 = net.predict(im1)
    #overlap_results1 = 0.5 * im1 + 0.5 * results1[0]
    #vis_im1 = np.concatenate([im1 / 255.0, results1[0] / 255.0, overlap_results1 / 255.0], axis=1)

    results1=results1[0][:,:,0]*255
    plt.subplot(131)
    plt.imshow(im1)
    plt.subplot(132)
    plt.imshow(im2, cmap='gray')
    plt.subplot(133)
    plt.imshow(results1, cmap='gray')

    plt.show()


    # for i in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):
    #     _, res, out = net.sess.run([update_op, pred, net.output])
    #     save_pred_to_image(res=res,
    #                        shape=cfg.param['eval_size'],
    #                        save_path=os.path.dirname(cfg.model_paths['others']) + '/eval_img',
    #                        save_name='eval_%d_img.png' % i)
    #
    # final_mIou = net.sess.run(mIoU)
    #
    # print('mIoU: {}'.format(final_mIou))
    #
    # Config.save_to_json(dict={'FINAL_MIOU': float(final_mIou), "EVAL_STEPS": cfg.param['eval_steps']},
    #                     path=os.path.dirname(cfg.model_paths['others']),
    #                     file_name='eval.json')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(model_log_dir='2018-11-07_00-57-50_', check_point=2299)

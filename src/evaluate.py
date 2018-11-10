import argparse

import tensorflow as tf
from tqdm import trange
from utils.image_reader import _image_mirroring,_random_crop_and_pad_image_and_labels,_image_scaling
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

    if cfg.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'] + 1)
    elif cfg.dataset == 'cityscapes':
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    elif cfg.dataset == 'others':
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])

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

    duration=0
    for i in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):

        start=time.time()
        _, res, input,labels,out = net.sess.run([update_op, pred, net.images,net.labels,net.output])
        end=time.time()

        duration+=(end-start)
        # if i % 100==0:
        #
        #     save_pred_to_image(res=res,
        #                        shape=cfg.param['eval_size'],
        #                        save_path=os.path.dirname(cfg.model_paths['others']) + '/eval_img',
        #                        save_name='eval_%d_img.png' % i)


        if i % 100 == 0:

            save_pred_to_image(res=res,
                               shape=cfg.param['eval_size'],
                               save_path=os.path.dirname(cfg.model_paths['others']) + '/eval_img',
                               save_name='eval_%d_img.png' % i)

            input = np.squeeze(input)
            n_input = _extract_mean_revert(input, IMG_MEAN, swap_channel=True)
            n_input = n_input.astype(np.uint8)
            input_image = Image.fromarray(n_input, 'RGB')


            res = np.array(np.reshape(res, cfg.param['eval_size']), dtype=np.uint8) * 255
            res = Image.fromarray(res.astype(np.uint8))
            labels = np.squeeze(labels) * 255
            labels = Image.fromarray(labels.astype(np.uint8))
            fig, ax1 = plt.subplots(figsize=(58, 13))



            plt.subplot(131)
            plt.imshow(input_image)
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(labels, cmap='gray')
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(res, cmap='gray')
            plt.axis('off')

            save_comparation_path = os.path.dirname(cfg.model_paths['others']) + '/eval_compare'
            if os.path.exists(save_comparation_path) is False:
                os.mkdir(save_comparation_path)
            plt.savefig(os.path.join(save_comparation_path, 'eval_%d_img.png' % i))


    final_mIou = net.sess.run(mIoU)
    print('total time:{} mean inference time:{} mIoU: {}'.format(duration,duration/cfg.param['eval_steps'],final_mIou))

    Config.save_to_json(dict={'FINAL_MIOU': float(final_mIou), "EVAL_STEPS": cfg.param['eval_steps']},
                        path=os.path.dirname(cfg.model_paths['others']),
                        file_name='eval.json')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main(model_log_dir='2018-11-08_13-21-26_restore_nonaug', check_point=19)


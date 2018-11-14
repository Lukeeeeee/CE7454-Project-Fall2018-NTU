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
import torch
import pandas
import glob

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


def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    mask[0] = 0
    mask[-1] = 0
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    if len(runs) % 2 != 0:
        runs = list(runs)
        runs.append(len(inds) - runs[-1])
        assert runs[-1] <= len(inds)
    rle = ' '.join([str(r) for r in runs])
    return rle


def main(model_log_dir, check_point, mode):
    tf.reset_default_graph()
    args = get_arguments()

    print('mode:{}'.format(mode))
    if mode == 'eval' or mode == 'compute_speed':

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

        duration = 0
        if mode == 'eval':

            for i in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):
                start = time.time()
                _, res, input, labels, out = net.sess.run([update_op, pred, net.images, net.labels, net.output])
                end = time.time()

                duration += (end - start)

                input = np.squeeze(input)
                n_input = _extract_mean_revert(input, IMG_MEAN, swap_channel=True)
                n_input = n_input.astype(np.uint8)
                input_image = Image.fromarray(n_input, 'RGB')

            final_mIou = net.sess.run(mIoU)
            print('total time:{} mean inference time:{} mIoU: {}'.format(duration, duration / cfg.param['eval_steps'],
                                                                         final_mIou))

            Config.save_to_json(dict={'FINAL_MIOU': float(final_mIou), "EVAL_STEPS": cfg.param['eval_steps']},
                                path=os.path.dirname(cfg.model_paths['others']),
                                file_name='eval.json', mode=mode)
        else:
            for i in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):
                start = time.time()
                res = net.sess.run(pred)
                end = time.time()
                duration += (end - start)

            print('total time:{} mean inference time:{}'.format(duration, duration / cfg.param['eval_steps']))
            Config.save_to_json(dict={'Total Inference Time': float(duration),
                                      "Mean Inference Time": duration / cfg.param['eval_steps']},
                                path=os.path.dirname(cfg.model_paths['others']),
                                file_name='eval.json', mode=mode)
    else:
        '''inference mode'''
        args = get_arguments()
        cfg = Config(dataset=args.dataset,
                     is_training=False,
                     filter_scale=args.filter_scale,
                     eval_path_log=os.path.join(LOG_PATH, model_log_dir))
        cfg.model_paths['others'] = os.path.join(LOG_PATH, model_log_dir, 'model.ckpt-%d' % check_point)

        model = model_config[args.model]

        reader = ImageReader(cfg=cfg, mode='eval')
        net = model(cfg=cfg, mode='inference')

        net.create_session()
        net.restore(cfg.model_paths[args.model])

        dir = '/home/dls/meng/DLProject/kaggle_dataset/dataset/test/'
        N = len(list(os.listdir(dir)))
        raw_data = {'img': [], 'rle_mask': []}
        # f.write('img,rle_mask\n')
        duration = 0
        fig_list = glob.glob(dir + '*.jpg')
        for index, i in zip(range(len(fig_list)), fig_list):
            img = Image.open(i)

            start = time.time()

            icnet_predict = net.predict(img)

            stop = time.time()

            duration += (stop - start)
            mask_array = np.squeeze(icnet_predict)

            en = run_length_encode(mask_array)
            print('{}/{} cost:{}s'.format(index, N, str(stop - start)))
            if i.find('.jpg') != -1:
                print('i is {}'.format(i))
                # f.write('{},{}\n'.format(i, en))
                # f.flush()
                raw_data['img'].append(os.path.basename(i))
                raw_data['rle_mask'].append(en)
            else:
                print('i is {}, not .jpg, exit now!'.format(i))
                exit()

        mean_inference_time = duration / (index + 1)
        df = pandas.DataFrame(raw_data, columns=['img', 'rle_mask'])
        with open(os.path.join(LOG_PATH, model_log_dir, 'SUBMISSION.csv'), mode='w') as f:
            df.to_csv(f, index=False)

        Config.save_to_json(
            dict={'Total Inference Time': float(duration), "Mean Inference Time": mean_inference_time},
            path=os.path.dirname(cfg.model_paths['others']),
            file_name='inference.json', mode=mode)
        sess = tf.get_default_session()
        if sess:
            sess._exit__(None, None, None)


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of the
    # original image) by setting those pixels to '0' explicitly. We do not
    # expect these to be non-zero for an accurate mask, so should not
    # harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def recover_result(res_list):
    # res_list = res_list

    new_res = []
    new_list = []

    len_res = len(res_list)
    for j in range(len_res):
        print("%d / %d" % (j, len_res))
        re = res_list[j]
        re = re.decode('ascii')
        re = re.split(' ')
        for i in range(len(re)):
            re[i] = int(re[i])
        new_list.append(re)

    size = [1280, 1918]
    full_length = size[0] * size[1] - 1
    for ori_res in new_list:
        res = np.zeros(shape=size)
        res = np.reshape(res, [-1])
        if len(ori_res) % 2 != 0:
            ori_res.append(full_length - ori_res[-1])
        assert len(ori_res) % 2 == 0
        assert ori_res[-1] <= full_length
        # for i in range(0, len(ori_res), 2):
        #     index = ori_res[i]
        #     length = ori_res[i + 1]
        #     res[index: index + length] = 1
        # res[0] = 0
        # res[-1] = 0
        ori_res = ' '.join([str(r) for r in ori_res])
        new_res.append(ori_res)
    return new_res


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #
    # main(model_log_dir='2018-11-13_16-29-55_v2_best_hyper_parameter_epoch_200_continue',
    #      check_point=44,
    #      mode='inference')
    # main(model_log_dir='2018-11-12_22-31-51_v2_best_hyper_parameter_epoch_20',
    #      check_point=19,
    #      mode='inference')
    path = '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/submit_csv/he/SUBMISSION.csv'
    # res = pandas.read_csv(filepath_or_buffer=path)

    # print(len(res))
    import numpy as np

    res = np.genfromtxt(path, delimiter=',', dtype=None)
    imag_list = res[1:, 0]
    res_list = res[1:, 1]
    imag_list = [i.decode('ascii') for i in imag_list]

    count = 0
    raw_data = {'img': [], 'rle_mask': []}
    new_res = recover_result(res_list=res_list)
    raw_data['img'] = imag_list
    raw_data['rle_mask'] = new_res
    # for res, i, index in zip(new_res, imag_list, range(len(new_res))):
    #     print("%d / %d" % (index, len(new_res)))
    #     # mask_array = np.squeeze(new_res)
    #     # en = run_length_encode(mask_array)
    #     raw_data['img'].append(str(i))
    #     raw_data['rle_mask'].append(res)

    df = pandas.DataFrame(raw_data, columns=['img', 'rle_mask'])
    with open('./SUBMISSION_he_tar.csv', mode='w') as f:
        df.to_csv(f, index=False)

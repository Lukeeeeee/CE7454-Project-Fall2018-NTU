import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
PAR_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
sys.path.append(PAR_PATH)
from src.model import ICNet, ICNet_BN
import os
from log import LOG_PATH
import argparse
import tensorflow as tf
import numpy as np
import time
from PIL import Image
# from test.TernausNet import Example
from src.TernausNet.Example import get_model, ternauNet
from utils.config import Config
import glob
from src.submit import run_length_encode
import pandas
from inference import load_single_image, INFER_SIZE

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


def main(model_log_dir, check_point, test_data_dir):
    tf.reset_default_graph()

    args = get_arguments()

    cfg = Config(dataset=args.dataset,
                 is_training=False,
                 INFER_SIZE=INFER_SIZE,
                 filter_scale=args.filter_scale,
                 eval_path_log=os.path.join(LOG_PATH, model_log_dir))
    cfg.model_paths['others'] = os.path.join(LOG_PATH, model_log_dir, 'model.ckpt-%d' % check_point)

    model = model_config[args.model]

    net = model(cfg=cfg, mode='inference')

    weight = [0.4, 0.6]
    ensemble_input = tf.placeholder(dtype=net.logits_up.dtype, shape=[None])

    ensemble_pred = tf.split(net.logits_up, 2, axis=len(net.logits_up.get_shape()) - 1)[1] * weight[0]
    ensemble_pred = tf.reshape(ensemble_pred, [-1, ])
    ensemble_pred = ensemble_pred + tf.cast(ensemble_input, tf.float32) * tf.constant(weight[1])
    ensemble_pred = tf.round(ensemble_pred)

    assert cfg.dataset == 'others'

    raw_data = {'img': [], 'rle_mask': []}

    net.create_session()
    net.restore(cfg.model_paths[args.model])
    duration = 0

    fig_list = glob.glob(test_data_dir + '*.jpg')
    t_model = get_model()

    for index, i in zip(range(len(fig_list)), fig_list):
        img = Image.open(i)
        start = time.time()

        input = np.squeeze(np.array(img, dtype=np.float32))
        n_input = input.astype(np.uint8)
        res2, tnet_mask = ternauNet(n_input, t_model)


        feed_dict = {ensemble_input: np.reshape(res2, [-1, ]),
                     net.img_placeholder: load_single_image(img_path=i, cfg=cfg)}

        ensemble_result, icnet = net.sess.run([ensemble_pred, net.output], feed_dict=feed_dict)

        end = time.time()
        duration += (end - start)

        # results = cv2.cvtColor((np.reshape(np.array(ensemble_result), cfg.param['eval_size']) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # tnet = Image.fromarray((tnet_mask * 255).astype(np.uint8))
        #
        # icnet = np.array(np.reshape(icnet, cfg.param['eval_size']), dtype=np.uint8) * 255
        # icnet = Image.fromarray(icnet.astype(np.uint8))
        #
        # fig, ax1 = plt.subplots(figsize=(5, 3))
        #
        # plot1 = plt.subplot(141)
        # plot1.set_title("Original", fontsize=10)
        # plt.imshow(img)
        # plt.axis('off')
        #
        # plot1 = plt.subplot(142)
        # plot1.set_title("TernauNet", fontsize=10)
        # plt.imshow(tnet)
        # plt.axis('off')
        #
        # plot1 = plt.subplot(143)
        # plot1.set_title("ICnet", fontsize=10)
        # plt.imshow(icnet)
        # plt.axis('off')
        #
        # plot2 = plt.subplot(144)
        # plot2.set_title("Ensemble Result", fontsize=10)
        # plt.imshow(results, cmap='gray')
        # plt.axis('off')
        # plt.show()

        ensemble_result = np.squeeze(ensemble_result)
        en = run_length_encode(ensemble_result)
        if i.find('.jpg') != -1:
            print('{}/ {} i is {}'.format(index, len(fig_list), i))
            raw_data['img'].append(os.path.basename(i))
            raw_data['rle_mask'].append(en)
        else:
            print('i is {}, not .jpg, exit now!'.format(i))
            exit()

    mean_inference_time = duration / (len(fig_list) + 1)

    df = pandas.DataFrame(raw_data, columns=['img', 'rle_mask'])
    with open(os.path.join(LOG_PATH, model_log_dir, 'SUBMISSION.csv'), mode='w') as f:
        df.to_csv(f, index=False)
    Config.save_to_json(
        dict={'Total Inference Time': float(duration), "Mean Inference Time": mean_inference_time},
        path=os.path.dirname(cfg.model_paths['others']),
        file_name='ensemble_inference.json', mode='eval')

    sess = tf.get_default_session()
    if sess:
        sess._exit__(None, None, None)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    main(model_log_dir='2018-11-08_08-56-52__v2_DEFAULT_CONFIG_LR_0.000500',
         check_point=19,
         test_data_dir='/home/dls/meng/DLProject/kaggle_carvana_segmentation/dataset/test/')

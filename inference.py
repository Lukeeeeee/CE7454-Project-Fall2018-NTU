import os

from log import LOG_PATH
from src.model import ICNet, ICNet_BN

'''add'''
import argparse
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
from utils.config import Config

# from test.TernausNet import Example

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}
IMG_MEAN = np.array((177.682, 175.84, 174.21), dtype=np.float32)
INFER_SIZE = [1280, 1918, 3]


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


def load_single_image(img_path, cfg):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape != cfg.INFER_SIZE:
        img = cv2.resize(img, (cfg.INFER_SIZE[1], cfg.INFER_SIZE[0]))

    return img


def main(model_log_dir, check_point, mode, img_path=None, testdir=None, submit=False):
    print('mode:{}'.format(mode))

    N = len(list(os.listdir(testdir)))
    '''inference mode'''
    args = get_arguments()
    cfg = Config(dataset=args.dataset,
                 is_training=False,
                 filter_scale=args.filter_scale,
                 eval_path_log=os.path.join(LOG_PATH, model_log_dir), INFER_SIZE=INFER_SIZE)
    cfg.model_paths['others'] = os.path.join(LOG_PATH, model_log_dir, 'model.ckpt-%d' % check_point)

    model = model_config[args.model]

    net = model(cfg=cfg, mode='inference')

    net.create_session()
    net.restore(cfg.model_paths[args.model])

    if not submit:
        image1 = load_single_image(img_path, cfg)
        results1 = net.predict(image1)
        results1 = np.squeeze(results1)

        results = cv2.cvtColor((results1 * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        '''Comparison'''

        overlap_results1 = 0.5 * image1 + 0.5 * results

        plt.figure(figsize=(60, 15))

        plot1 = plt.subplot(131)

        plot1.set_title("Input Image", fontsize=50)
        plt.imshow(image1)
        plt.axis('off')

        plot2 = plt.subplot(132)
        plot2.set_title("Predicted Mask", fontsize=50)
        plt.imshow(results)
        plt.axis('off')

        plot3 = plt.subplot(133)
        plot3.set_title("Mask over image", fontsize=50)
        plt.imshow(overlap_results1 / 255.0)
        plt.axis('off')

        plt.show()

    else:
        with open('SUBMISSION.csv', 'w') as f:
            f.write('img,rle_mask\n')
            duration = 0
            for index, i in enumerate(os.listdir(testdir)):
                img = Image.open(os.path.join(testdir, i))

                start = time.time()

                icnet_predict = net.predict(img)

                stop = time.time()

                duration += (stop - start)
                mask_array = np.squeeze(icnet_predict)

                plt.imshow(mask_array)
                plt.show()
                en = run_length_encode(mask_array)
                print('{}/{} cost:{}s'.format(index, N, str(stop - start)))
                if i.find('.jpg') != -1:
                    print('i is {}'.format(i))
                    f.write('{},{}\n'.format(i, en))
                    f.flush()
                else:
                    print('i is {}, not .jpg, exit now!'.format(i))
                    exit()

            mean_inference_time = duration / (index + 1)

        Config.save_to_json(
            dict={'Test set size': N, 'Test set path': testdir, 'Total Inference Time': float(duration),
                  "Mean Inference Time": mean_inference_time},
            path=os.path.dirname(cfg.model_paths['others']),
            file_name='inference.json', mode=mode)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    dir = '/media/data1/hewei/test/'
    test_image = '/home/wei005/PycharmProjects/CE7454_Project_Fall2018_NTU/data/Kaggle/valid/data/0ce66b539f52_01.jpg'
    main(model_log_dir='2018-11-08_13-21-26_restore_nonaug', check_point=19, mode='inference', img_path=test_image,
         testdir=dir, submit=False
         )

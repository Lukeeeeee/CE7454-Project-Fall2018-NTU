import cv2
import os
import torch
from src.TernausNet.unet_models import unet11
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
from time import time

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = unet11(pretrained='carvana')
    model.eval()
    return model.to(device)


def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


def crop_image(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]


def load_image(path, pad=True):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = path

    if not pad:
        return img

    height, width, _ = img.shape

    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def submit():
    """Used for Kaggle submission: predicts and encode all test images"""
    dir = '/media/data1/hewei/test/'
    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = get_model()

    N = len(list(os.listdir(dir)))

    with open('SUBMISSION.csv', 'a') as f:
        f.write('img,rle_mask\n')
        for index, i in enumerate(os.listdir(dir)):
            # img = Image.open(dir + i)
            img, pads = load_image(dir + i, pad=True)
            start = time()
            with torch.no_grad():
                input_img = torch.unsqueeze(img_transform(img).to(device), dim=0)
                mask = F.sigmoid(model(input_img))

            stop = time()
            mask_array = mask.data[0].cpu().numpy()[0]

            mask_array = crop_image(mask_array, pads)
            mask_array[mask_array > 0.5] = 1
            mask_array[mask_array <= 0.5] = 0

            plt.imshow(mask_array, cmap='gray')
            plt.show()

            en = run_length_encode(mask_array)
            print('{}/{} cost{}s'.format(index, N, str(stop - start)))
            # f.write('{},{}\n'.format(i, en))


def ternauNet(img, model):
    img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # TODO instead of load model every time, pass the model into as an argument.
    # model = unet11(pretrained='carvana')
    # model.eval()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = model.to(device)

    r, g, b = cv2.split(img)
    img_bgr = cv2.merge([b, g, r])

    img_bgr, pads = load_image(img_bgr, pad=True)

    with torch.no_grad():
        input_img = torch.unsqueeze(img_transform(img_bgr).to(device), dim=0)
        mask = F.sigmoid(model(input_img))

    mask_array = mask.data[0].cpu().numpy()[0]

    mask_array = crop_image(mask_array, pads)
    mask = mask_array.copy()

    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return mask_array, mask


if __name__ == '__main__':
    # net = UNet(3, 1).cuda()
    # net.load_state_dict(torch.load('MODEL.pth'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    submit()

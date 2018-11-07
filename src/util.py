import tensorflow as tf
from utils.config import Config
from src.model import ICNet_BN
from utils.config import Config
from utils.image_reader import ImageReader, prepare_label
from PIL import Image
import os
import numpy as np


def save_pred_to_image(res, shape, save_path, save_name):
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    res = np.array(np.reshape(res, shape), dtype=np.uint8) * 255
    img = Image.fromarray(res.astype(np.uint8), mode='L')
    img.save(os.path.join(save_path, save_name))



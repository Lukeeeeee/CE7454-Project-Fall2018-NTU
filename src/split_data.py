import os
import shutil

import pandas as pd
from PIL import Image

from data import DATA_PATH

f = open("../valid.csv", "rb")
list = pd.read_csv(f)

data_destination = os.path.join(DATA_PATH, 'Kaggle/valid/data')
mask_destination = os.path.join(DATA_PATH, 'Kaggle/valid/mask')

if not os.path.exists(data_destination):
    os.makedirs(data_destination)

if not os.path.exists(mask_destination):
    os.makedirs(mask_destination)

data = list['path'].tolist()
masks = list['mask_path'].tolist()

'''Copy files to this project'''
for images in data:
    shutil.copy(images, data_destination)

for mask in masks:
    im = Image.open(mask)
    image_name = mask.split('/')[-1]
    image_name = image_name.replace('gif', 'png')
    im.save(os.path.join(mask_destination, image_name))
    # shutil.copy(mask,mask_destination)

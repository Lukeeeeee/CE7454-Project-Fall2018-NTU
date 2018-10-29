import os
import pandas as pd
import shutil

f=open("test.csv","rb")
list=pd.read_csv(f)
data_destination='/home/wei005/PycharmProjects/ICNet-tensorflow/data/Kaggle/Test/data/'
mask_destination='/home/wei005/PycharmProjects/ICNet-tensorflow/data/Kaggle/Test/mask/'
data=list['path'].tolist()
masks=list['mask_path'].tolist()
for images in data:
    shutil.copy(images,data_destination)

for mask in masks:
    shutil.copy(mask,mask_destination)
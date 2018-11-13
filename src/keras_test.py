import keras
from keras.applications.vgg16 import VGG16

model=VGG16(include_top=True)
model.summary()
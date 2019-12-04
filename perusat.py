#Dataset perusat imagenes 512x512
# normalizacion perusat_v3
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
#import tifffile
import cv2
import numpy
from torchvision import transforms


def my_transform_go(x):
    imx = x.transpose((2, 0, 1))
    return imx

def my_transform_back(x):
    numpyx = numpy.array(x)
    numpyx = numpyx.transpose((1, 2, 0))
    return numpyx

def my_transform_tensor(x):
    tensor = torch.FloatTensor(x)
    return tensor

def my_transform_resize(x):
    scale_percent = 25  # percent of original size
    width = int(x.shape[1] * scale_percent / 100)
    height = int(x.shape[0] * scale_percent / 100)
    dim = (width, height)
    res_image = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    return res_image

# Se redimensiona a 128x128
def my_transform_128(x):
    res_image = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
    return res_image

# Se corta la imagen de acuerdo a la escala
def my_transform_crop(x, scale=1.0):
    center_x, center_y = x.shape[1] / 2, x.shape[0] / 2
    width_scaled, height_scaled = x.shape[1] * scale, x.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def my_transform_512(x):
    res_image = cv2.resize(x, (512, 512), interpolation=cv2.INTER_CUBIC)
    return res_image

#Normalizacion zscore
def my_transform_nor(x):
    nor_image = (x - mean) / std
    return nor_image

def my_transform_norb(x):
    nor_image = (x - mean_tensor) / std_tensor
    return nor_image

#Salida entre -1 y 1
def my_transform_nor2(x):
    nor_image = (x - x.mean())
    nor_image = nor_image/nor_image.max()
    return nor_image

#Salida entre 0 y 1 , min max
def my_transform_nor3(x):
    nor_image = (x - x.min()) / (x.max()-x.min())
    return nor_image

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        #Transformacion para obtener una imagen en LR 128x128 - interpolación
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose([
                            transforms.Lambda(lambda x: my_transform_crop(x, 0.25)),
                            transforms.Lambda(lambda x: my_transform_resize(x)),
                            #transforms.Lambda(lambda x: my_transform_128(x)),
                            #transforms.Lambda(lambda x: my_transform_nor(x)),
                            transforms.Lambda(lambda x: my_transform_go(x)),
                            transforms.Lambda(lambda x: my_transform_tensor(x))
                            ])
        #Transformacion para obtener una imagen en HR 512x512 - interpolación
        self.hr_transform = transforms.Compose([
                            transforms.Lambda(lambda x: my_transform_crop(x, 0.25)),
                            #transforms.Lambda(lambda x: my_transform_512(x)),
                            #transforms.Lambda(lambda x: my_transform_nor(x)),
                            transforms.Lambda(lambda x: my_transform_go(x)),
                            transforms.Lambda(lambda x: my_transform_tensor(x))
                            ])

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        # Prueba cambio de fuente
        #img_hr = self.img

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)




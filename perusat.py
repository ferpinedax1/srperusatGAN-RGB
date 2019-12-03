#Dataset perusat imagenes 512x512
# normalizacion perusat_v3
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import cv2
import numpy
from torchvision import transforms

#Valores de normalizacion - Perusat_v4
mean = np.array([373.5604, 370.5355, 412.6234])
std = np.array([117.6282,  75.4106,  61.8215])

mean_tensor = torch.FloatTensor(mean)
std_tensor = torch.FloatTensor(std)

#Transformaciones lambda - soporte para tif
#Perusat tamaño fijo de 512x512
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

def my_transform_128(x):
    res_image = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
    return res_image

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
                            transforms.Lambda(lambda x: my_transform_128(x)),
                            transforms.Lambda(lambda x: my_transform_nor(x)),
                            transforms.Lambda(lambda x: my_transform_go(x)),
                            transforms.Lambda(lambda x: my_transform_tensor(x))
                            ])
        #Transformacion para obtener una imagen en HR 512x512 - interpolación
        self.hr_transform = transforms.Compose([
                            transforms.Lambda(lambda x: my_transform_512(x)),
                            transforms.Lambda(lambda x: my_transform_nor(x)),
                            transforms.Lambda(lambda x: my_transform_go(x)),
                            transforms.Lambda(lambda x: my_transform_tensor(x))
                            ])

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = tifffile.imread(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)




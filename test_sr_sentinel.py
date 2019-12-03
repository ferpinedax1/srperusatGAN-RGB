# Normalizacion perusat_v4
# Prueba SR perusat

from model_BN import Generador
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import tifffile
import numpy as np
import cv2
from perusat import *
import torch.nn as nn

# Seleccion de GPU a utilizar, si se cuenta con 1 GPU, valor 0
GPU_use = 0

#Cargar archivo pre entrenado, BN o SN
checkpoint_model = "/home/fpineda/test/Tesis/input_prueba_srperusat/3bandas/BN/generator_49.pth"

#Archivo para aplicar SR(512x512, 3bandas, tif)
image_path = "/home/fpineda/dataset/sentinel_validation/sentinel3bandas/1_tile_1024-1536.tif"

# Carpetas usadas en el test
os.makedirs("sr_test", exist_ok=True)
os.makedirs("ssim", exist_ok=True)

# GPU
torch.cuda.set_device(GPU_use)
device = torch.device("cuda")

# Al generador con un upscale de 4
upscale = 4

# Bandas a utilizar
channel = 3

generator = Generador(upscale,channel).to(device)
generator.load_state_dict(torch.load(checkpoint_model))
generator.eval()

# Normalizacion de la imagen origen sentinel
# En el dataset con solo 3 bandas, se cambiaron la bande 1 x banda 3
# Por eso para normalizar se debe cambiar los valores
#mean = np.array([1272.3330, 1240.9513, 1301.6351, 2091.1609])
#std = np.array([186.4024, 237.6374, 380.6278, 432.6853])
mean = np.array([1302.5645, 1241.7332, 1272.9674])
std = np.array([380.4034, 237.3247, 186.0898])


# Transformaciones a la imagen origen
def my_transform_nor(x):
    nor_image = (x - mean) / std
    return nor_image


def my_transform_tensor(x):
    tensor = torch.FloatTensor(x)
    return tensor


def my_transform_128(x):
    res_image = cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)
    return res_image


def my_transform_go(x):
    imx = x.transpose((2, 0, 1))
    return imx


transform = transforms.Compose([
    #transforms.Lambda(lambda x: my_transform_128(x)),
    transforms.Lambda(lambda x: my_transform_nor(x)),
    transforms.Lambda(lambda x: my_transform_go(x)),
    transforms.Lambda(lambda x: my_transform_tensor(x))
])

transform_png = transforms.Compose([
    transforms.Lambda(lambda x: my_transform_nor(x)),
    transforms.Lambda(lambda x: my_transform_go(x)),
    transforms.Lambda(lambda x: my_transform_tensor(x))
])

# Archivo de entrada para aplicar SR
# Interpolacion a 128x128
# Guardo como png la imagen origina, downscale y en super resolucion
# A estas imagenes aplico SSIM
# Tambien se guarda las imagenes en TIF en lR y en SR
# Se guardan como tensores, falta añadir el extractor y guardado en tif
imagen_tif = tifffile.imread(image_path)

# La imagen tif fuente la guardo como png para aplicar metricas
imagen_png_source = transform_png(imagen_tif)
save_image(imagen_png_source, "ssim/png_source.png", normalize=True)

# La imagen tif downscale a 128 para aplicar SR
imagen = transform(imagen_tif)
print(imagen.shape)
torch.save(imagen, "sr_test/raw_lr")

image_tensor = Variable(imagen).to(device).unsqueeze(0)
image_tensor2 = torch.tensor(image_tensor, device=device).float()

# Guardo la  imagen en baja resolucion como  png
save_image(image_tensor2, "ssim/png_lr.png", normalize=True)

# Aplico SR y guardo la imagen como PNG y raw
sr_image = generator(image_tensor2)
torch.save(sr_image, "sr_test/raw_sr")
save_image(sr_image, "ssim/png_sr.png", normalize=True)

# La iamgen en SR de 2048 le hago downscale a 512 para usar las metricas
sr_image_512 = nn.functional.interpolate(sr_image, scale_factor=0.25, mode='bicubic')
save_image(sr_image_512, "ssim/png_sr_512.png", normalize=True)

# Guarda en tif el resutlado SR
load_lr = torch.load("sr_test/raw_lr")
load_lr = load_lr.cpu().data.numpy()
load_lr = load_lr.transpose((1, 2, 0))

load_sr = torch.load("sr_test/raw_sr")
load_sr = load_sr[0].cpu().data.numpy()
load_sr = load_sr.transpose((1, 2, 0))

def un_normalize(array):
    mean = np.array([1272.3330, 1240.9513, 1301.6351])
    std = np.array([186.4024, 237.6374, 380.6278])
    array = (array * std) + mean
    return array

tif_img_lr = un_normalize(load_lr)
tif_img_sr = un_normalize(load_sr)
tifffile.imsave("sr_test/tif_lr.tif", tif_img_lr)
tifffile.imsave("sr_test/tif_sr.tif", tif_img_sr)

#################
# SSIM
#################

from SSIM_PIL import compare_ssim
from PIL import Image

image1 = Image.open("ssim/png_source.png")
image2 = Image.open("ssim/png_sr_512.png")
valor_ssim = compare_ssim(image1, image2)
print("SSIM: ", valor_ssim)
print("#############################\n\n")

##########################
# Metricas
#########################
from sewar.full_ref import psnr
from sewar.full_ref import ssim
from sewar.full_ref import uqi
from sewar.full_ref import mse
from sewar.full_ref import rmse_sw
from sewar.full_ref import ergas
from sewar.full_ref import scc
from sewar.full_ref import rase
from sewar.full_ref import sam
from sewar.full_ref import msssim
from sewar.full_ref import vifp

img1 = cv2.imread("ssim/png_source.png")
img2 = cv2.imread("ssim/png_sr_512.png")

print("Metricas\n")

uqi = uqi(img1, img2)
print("uqi: ", uqi)

psnr = psnr(img1, img2)
print("psnr: ", psnr)

ssim = ssim(img1, img2)
print("ssim: ", ssim)

mse = mse(img1, img2)
print("mse: ", mse)

rmse_sw = rmse_sw(img1, img2)
# print("rmse_sw: ", rmse_sw)

ergas = ergas(img1, img2)
print("ergas: ", ergas)

scc = scc(img1, img2)
print("scc: ", scc)

rase = rase(img1, img2)
print("ergas: ", rase)

sam = sam(img1, img2)
print("sam: ", sam)

msssim = msssim(img1, img2)
print("msssim: ", msssim)

vifp = vifp(img1, img2)
print("vifp: ", vifp)






# Super resolucion Perusat-1
# Imagenes en formato tif
# Modelo corre solo con GPU
# Dataset perusat_v4 imagenes costa norte Peru

import os
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from model_BN import *
from perusat import *
import torch.nn as nn
import torch.nn.functional as F
import torch

# Directorios usados en el proyecto
os.makedirs("imagen_png", exist_ok=True)
os.makedirs("modelo_entrenado", exist_ok=True)
os.makedirs("imagen_tif", exist_ok=True)
os.makedirs("imagen_raw", exist_ok=True)

# Configuracion entrenamiento
epoch = 0
n_epochs = 50
batch_size = 4

# Seleccion de GPU a utilizar, si se cuenta con 1 GPU, valor 0
GPU_use = 1

# Optimizacion
lr = 0.001
b1 = 0.9
b2 = 0.999

# Tamaño de la imagen perusat 1
hr_height = 512
hr_width = 512

# Bandas a utilizar
channel = 3

# Factor SR x 4
upscale = 4

# GPU
torch.cuda.set_device(GPU_use)
cuda = torch.cuda.is_available()

# Imagenes dataset Perusat 512x512
hr_shape = (hr_height, hr_width)

# Inicializa generador y discriminador
generator = Generador(upscale, channel)
discriminator = Discriminador(channel)
feature_extractor = FeatureExtractor()

# Feature extractor en modo inferencia
feature_extractor.eval()

# Función perdida, loss function
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

# Datos en GPU
generator = generator.cuda()
discriminator = discriminator.cuda()
feature_extractor = feature_extractor.cuda()
criterion_GAN = criterion_GAN.cuda()
criterion_content = criterion_content.cuda()

# Optimizador
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor

# Conjunto de entrenamiento perusat-1
# PC lenovo
#path = "/home/fpineda/Documentos/Gdrive/Colab Notebooks/Dataset/perusat_v1/TIFperusat3bandas_lite/"

# GPU enterprise
path = "/home/fpineda/dataset/perusat_v4/PNGperusat/"
#path = "/home/fpineda/dataset/TIFperusat3bandas_lite/"

# GPU oso
#path = "/home/fpineda"

# Dataloader, imagenes tif
dataloader = DataLoader(ImageDataset(path, hr_shape=hr_shape), batch_size=batch_size,
                        shuffle=True, num_workers=1,)

# --------------
#  Entrenamiento
# --------------

for epoch in range(epoch, n_epochs):
    for i, imgs in enumerate(dataloader):

        # Imagen de entrada en baja y alta resolución
        imgs_lr = imgs["lr"].type(Tensor)
        imgs_hr = imgs["hr"].type(Tensor)

        # Adversarial ground truths
        valid = torch.ones(imgs_lr.size(0)).type(Tensor)
        fake = torch.zeros(imgs_lr.size(0)).type(Tensor)

        # ------------------------
        #  Entrenamiento generador
        # ------------------------

        optimizer_G.zero_grad()

        # Se genera una imagen en SR a partir de la imagen en LR
        gen_sr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_sr), valid)

        # Content loss
        gen_features = feature_extractor(gen_sr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ----------------------------
        #  Entrenamiento discriminador
        # ----------------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images

        loss_D_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_D_fake = criterion_GAN(discriminator(gen_sr.detach()), fake)

        # Total loss
        loss_D = (loss_D_real + loss_D_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # ---------------------------
        #  Progreso del entrenamiento
        # ---------------------------

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, i, len(dataloader),
                                                                             loss_D.item(), loss_G.item()))



    # Guardo datos en lr y sr
    torch.save(imgs_lr, 'imagen_raw/raw_lr%d' % epoch)
    torch.save(gen_sr, 'imagen_raw/raw_sr%d' % epoch)

    # Guardo imagen original
    imgs_source = imgs_lr
    save_image(imgs_source, "imagen_png/png_source_%d.png" % epoch, normalize=True)

    # La imagen en lr se le hace interpolacion a 512, y comparar con SR
    imgs_inter = nn.functional.interpolate(imgs_source, scale_factor=4, mode='bicubic')
    #imgs_inter = cv2.resize(np.float32(imgs_source), (512, 512), interpolation=cv2.INTER_CUBIC)

    # Guardo una imagen en SR
    save_image(gen_sr, "imagen_png/png_sr_%d.png" % epoch, normalize=True)

    # Imagen interpolacion vs SR
    gen_sr = make_grid(gen_sr, nrow=1, normalize=True)
    imgs_inter = make_grid(imgs_inter, nrow=1, normalize=True)
    img_grid = torch.cat((imgs_inter, gen_sr), -1)

    # Guardo imagen por cada epoca
    save_image(img_grid, "imagen_png/png_perusat_%d.png" % epoch, normalize=True)

    # Guardo modelo entrenado por cada epoca
    torch.save(generator.state_dict(), "modelo_entrenado/generator_%d.pth" % epoch)
    torch.save(discriminator.state_dict(), "modelo_entrenado/discriminator_%d.pth" % epoch)

    # Guarda en tif un resultado por epoca

    #load_lr = torch.load("imagen_raw/raw_lr%d" % epoch)
    #load_lr = load_lr[0].cpu().data.numpy()
    #load_lr = load_lr.transpose()

    #load_sr = torch.load("imagen_raw/raw_sr%d" % epoch)
    #load_sr = load_sr[0].cpu().data.numpy()
    #load_sr = load_sr.transpose()

    #def un_normalize(array):
    #    mean = np.array([373.5604, 370.5355, 412.6234])
    #    std = np.array([117.6282, 75.4106, 61.8215])
    #    array = (array * std) + mean
    #    return array

    #tif_img_lr = un_normalize(load_lr)
    #tif_img_sr = un_normalize(load_sr)
    #tifffile.imsave("imagen_tif/tif_lr_%d.tif" % epoch, tif_img_lr)
    #tifffile.imsave("imagen_tif/tif_sr_%d.tif" % epoch, tif_img_sr)

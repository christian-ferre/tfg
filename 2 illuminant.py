import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import colour
from colour import CCT_to_xy
from colour import xy_to_CCT

ruta_principal = os.path.abspath("galaxy")

# Inicializamos una lista para las imagenes
images = []

# Cargamos las imagenes
for i in range(0, 10):
    nombre_carpeta = f"Place{i}"
    ruta = os.path.join(ruta_principal, nombre_carpeta)
    imagenes_carpeta = []
    for image in os.listdir(ruta):
        if image.endswith("_1.jpg") or image.endswith("_12.jpg"):
            ruta_imagen = os.path.join(ruta, image)
            imagen = cv2.imread(ruta_imagen)
            imagenes_carpeta.append(imagen)
    images.append(imagenes_carpeta)


def mccamy(list, pos, save=False):
    img = np.array(list[0])
    # Convertir de RGB a XYZ
    matrix_RGB_to_XYZ = colour.RGB_COLOURSPACES['sRGB'].RGB_to_XYZ
    illuminant = np.array([0.95047, 1.0, 1.08883])  # D65
    img_xyz = colour.RGB_to_XYZ(img / 255, colour.RGB_COLOURSPACES['sRGB'], illuminant, matrix_RGB_to_XYZ)

    # Convertir de XYZ a xy
    img_xy = colour.XYZ_to_xy(img_xyz)

    # Calcular la temperatura de color en Kelvin
    img_temp = colour.xy_to_CCT(img_xy)

    # Calcular los factores de corrección
    t1, t2 = 6500, img_temp
    r_corr = ((t2 / t1) ** 2.4) * 0.5
    b_corr = ((t1 / t2) ** 2.4) * 0.5

    # Aplicar los factores de corrección a cada canal
    img[:, :, 0] *= r_corr
    img[:, :, 2] *= b_corr

    # Limitar los valores de píxel a 255.
    img[img > 255] = 255

    # Convertir la imagen de vuelta a un formato entero de 8 bits
    img = img.astype(np.uint8)

    # Mostramos la imagen original y la corregida

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(list[0], cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen original")
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Imagen corregida")
    ax3.imshow(cv2.cvtColor(list[1], cv2.COLOR_BGR2RGB))
    ax3.set_title("Imagen groundtruth")

    # Crear carpeta si no existe
    if save:
        carpeta = "2illuminant/McCamy"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        ruta = f"2illuminant/McCamy/McCamy{pos}.png"
        plt.savefig(ruta)


def gray_edge(list, pos, save=False):
    img = np.array(list[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calcular el valor medio de los píxeles en la imagen en escala de grises
    gray_mean = cv2.mean(gray)[0]

    # Calcular la mediana de los valores de los píxeles
    gray_median = cv2.medianBlur(gray, 5)

    # Calcular el valor medio de los píxeles en la imagen suavizada
    smooth_mean = cv2.mean(gray_median)[0]

    # Calcular el factor de escala para ajustar la imagen
    scale_factor = smooth_mean / gray_mean

    # Aplicar el factor de escala a la imagen original
    balanced_img = cv2.convertScaleAbs(img, alpha=scale_factor, beta=0)

    # Mostramos la imagen original y la corregida

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(list[0], cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen original")
    ax2.imshow(cv2.cvtColor(balanced_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Imagen corregida")
    ax3.imshow(cv2.cvtColor(list[0], cv2.COLOR_BGR2RGB))
    ax3.set_title("Imagen groundtruth")

    # Crear carpeta si no existe
    if save:
        carpeta = "2illuminant/Gray-Edge"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        ruta = f"2illuminant/Gray-Edge/Gray-Edge{pos}.png"
        plt.savefig(ruta)


def auto_wb(list, pos, save=False):
    img = np.array(list[0])
    # Aplica el balance de blancos automático
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_lab = cv2.merge((l, a, b))
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    # Mostramos la imagen original y la corregida

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(list[0], cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen original")
    ax2.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax2.set_title("Imagen corregida")
    ax3.imshow(cv2.cvtColor(list[0], cv2.COLOR_BGR2RGB))
    ax3.set_title("Imagen groundtruth")

    # Crear carpeta si no existe
    if save:
        carpeta = "2illuminant/auto_wb"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        ruta = f"2illuminant/auto_wb/auto_wb{pos}.png"
        plt.savefig(ruta)


for i, img in enumerate(images):
    # mccamy(img, 5000, 8000, i, True)
    gray_edge(img, i, True)
    auto_wb(img, i, True)

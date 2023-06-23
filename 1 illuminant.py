import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

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


def gray_world(list, pos, save=False):
    img = list[0]
    # Calculamos la media de cada canal RGB
    B, G, R = cv2.split(img)
    avgB = np.average(B)
    avgG = np.average(G)
    avgR = np.average(R)

    # Calculamos el factor de corrección de cada canal
    grayValue = (avgB + avgG + avgR) / 3
    bCorr = grayValue / avgB
    gCorr = grayValue / avgG
    rCorr = grayValue / avgR

    # Aplicamos corrección a cada canal
    newB = cv2.multiply(B, bCorr)
    newG = cv2.multiply(G, gCorr)
    newR = cv2.multiply(R, rCorr)

    # Combinamos los canales
    gray_world_img = cv2.merge((newB, newG, newR))

    # Mostramos la imagen original y la corregida
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen original")
    ax2.imshow(cv2.cvtColor(gray_world_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Imagen corregida")
    ax3.imshow(cv2.cvtColor(list[1], cv2.COLOR_BGR2RGB))
    ax3.set_title("Imagen groundtruth")

    # Creamos la carpeta si no existe
    if save:
        carpeta = "1illuminant/GrayWorld"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        ruta = f"1illuminant/GrayWorld/GrayWorld{pos}.png"
        plt.savefig(ruta)


def white_patch(list, pos, save=False):
    B, G, R = cv2.split(list[0])
    # Obtener el valor máximo de intensidad de cada canal
    max_b = np.max(B)
    max_g = np.max(G)
    max_r = np.max(R)

    # Verificar si es necesario aplicar corrección
    if max_b == 255 and max_g == 255 and max_r == 255:
        white_patch_img = list[0]
    else:
        # Aplicamos corrección a cada canal
        newB = cv2.divide(B, max_b)
        newG = cv2.divide(G, max_g)
        newR = cv2.divide(R, max_r)

        # Combinamos los canales
        white_patch_img = cv2.merge((newB, newG, newR))

    # mostramos la imagen original y la corregida
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(list[0], cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen original")
    ax2.imshow(cv2.cvtColor(white_patch_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Imagen corregida")
    ax3.imshow(cv2.cvtColor(list[1], cv2.COLOR_BGR2RGB))
    ax3.set_title("Imagen groundtruth")

    # Creamos la carpeta si no existe
    if save:
        carpeta = "1illuminant/WhitePath"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        ruta = f"1illuminant/WhitePath/WhitePath{pos}.png"
        plt.savefig(ruta)


def max_rgb(list, pos, save=False):
    # Dividir la imagen en sus canales RGB
    B, G, R = cv2.split(list[0])

    # encontrar el valor máximo para cada canal
    max_value = max([B.max(), G.max(), R.max()])

    # calcular el factor de escala para cada canal
    scale_b = max_value / B.max()
    scale_g = max_value / G.max()
    scale_r = max_value / R.max()

    # escalar cada canal de acuerdo al factor de escala
    balanced_b = cv2.convertScaleAbs(B, alpha=scale_b)
    balanced_g = cv2.convertScaleAbs(G, alpha=scale_g)
    balanced_r = cv2.convertScaleAbs(R, alpha=scale_r)

    # Fusionamos los canales en una imagen
    img_maxrgb = cv2.merge([balanced_b, balanced_g, balanced_r])

    # Mostramos la imagen original y la corregida
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(list[0], cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen original")
    ax2.imshow(cv2.cvtColor(img_maxrgb, cv2.COLOR_BGR2RGB))
    ax2.set_title("Imagen corregida")
    ax3.imshow(cv2.cvtColor(list[1], cv2.COLOR_BGR2RGB))
    ax3.set_title("Imagen groundtruth")

    # Creamos la carpeta si no existe
    if save:
        carpeta = "1illuminant/MaxRGB"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        ruta = f"1illuminant/MaxRGB/MaxRGB{pos}.png"
        plt.savefig(ruta)


def ganancia_maxima(list, pos, save=False):
    # Dividir la imagen en sus canales RGB
    B, G, R = cv2.split(list[0])

    # Calcular la desviación estándar de cada canal de color
    std_dev_B = np.std(B)
    std_dev_G = np.std(G)
    std_dev_R = np.std(R)

    # Identificar el canal de color con la mayor desviación estándar.
    max_std_dev = max(std_dev_R, std_dev_G, std_dev_B)

    if max_std_dev == std_dev_R:
        channel = R
    elif max_std_dev == std_dev_G:
        channel = G
    else:
        channel = B

    # Calcular la ganancia de cada canal dividiendo el valor máximo del canal por su valor medio.
    gain = np.max(channel) / np.mean(channel)

    # Aplicar la ganancia calculada a cada canal de color.
    R_adjusted = np.uint8(np.clip(R * gain, 0, 255))
    G_adjusted = np.uint8(np.clip(G * gain, 0, 255))
    B_adjusted = np.uint8(np.clip(B * gain, 0, 255))

    # Combinar los canales de color ajustados para formar la imagen final.
    img_maxganancia = cv2.merge((B_adjusted, G_adjusted, R_adjusted))

    # Mostramos la imagen original y la corregida
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(list[0], cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen original")
    ax2.imshow(cv2.cvtColor(img_maxganancia, cv2.COLOR_BGR2RGB))
    ax2.set_title("Imagen corregida")
    ax3.imshow(cv2.cvtColor(list[1], cv2.COLOR_BGR2RGB))
    ax3.set_title("Imagen groundtruth")

    # Creamos la carpeta si no existe
    if save:
        carpeta = "1illuminant/GananciaMaxima"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        ruta = f"1illuminant/GananciaMaxima/GananciaMaxima{pos}.png"
        plt.savefig(ruta)


for i, img in enumerate(images):
    gray_world(img, i, True)
    white_patch(img, i, True)
    max_rgb(img, i, True)
    ganancia_maxima(img, i, True)

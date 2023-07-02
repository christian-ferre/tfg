from tfg import Illuminant_estimation
import cv2
import matplotlib.pyplot as plt
import os

test = Illuminant_estimation()


def imagen_comparativa():
    width, height = 500, 500
    space_width = 50

    # Cargar la imagen original y redimensionarla
    image = cv2.imread("galaxy/Place9/Place9_1.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))

    # Generar las cinco subimágenes
    subimages = []
    labels = ['L1', 'L2', 'L3', 'L4', 'L5']

    # Ruta de la carpeta que contiene las imágenes TIFF
    data = 'Resultados/CorrectedMoment'
    archivos = os.listdir(data)
    archivos.reverse()
    for i, archivo_tiff in enumerate(archivos):
        ruta_imagen = os.path.join(data, archivo_tiff)
        imagen = cv2.imread(ruta_imagen)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        subimages.append(imagen)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calcular los límites de los ejes
    total_width = width + space_width + width * 5

    # Mostrar la imagen original en la posición deseada
    ax.imshow(image, extent=(0, width, 0, height))
    ax.text(width / 2, height + 20, 'Imagen original', color='black', fontsize=12, ha='center')

    # Mostrar las subimágenes con sus etiquetas
    for i, subimage in enumerate(subimages):
        x_start = width + space_width + i * width
        x_end = x_start + width
        ax.imshow(subimage, extent=(x_start, x_end, 0, height))
        ax.text(x_start + width / 2, height + 20, labels[i], color='black', fontsize=12, ha='center')

    # Configurar los límites y la apariencia de los ejes
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, height)
    ax.axis('off')

    # Guardar la imagen
    plt.savefig('imagen_comparativa.png', dpi=300, bbox_inches='tight')

    # Mostrar la imagen en pantalla (opcional)
    plt.show()


imagen_comparativa()

import cv2
import csv
import os

def save_points_to_csv(points, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(points)

def mark_rectangle(img):
    # Cargar la imagen
    clone = img.copy()

    # Lista para almacenar los puntos
    points = []

    # Función de retroalimentación para el evento del ratón
    def mouse_callback(event, x, y, flags, param):
        nonlocal clone
        if event == cv2.EVENT_LBUTTONDOWN:
            # Guardar las coordenadas del punto
            points.append((x, y))
            # Dibujar un círculo en el punto marcado
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('image', clone)
            # Cerrar la ventana si se han marcado los 2 puntos
            if len(points) == 2:
                cv2.destroyAllWindows()

    # Crear una ventana y establecer la función de retroalimentación del ratón
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', mouse_callback)

    # Mostrar la imagen original
    cv2.imshow('image', img)

    # Esperar a que se marquen los 2 puntos y se cierre la ventana
    cv2.waitKey(0)

    return points

# Lista para almacenar los puntos de todas las imágenes
all_points = []
carpeta = "raw_images_2illuminants"
# Obtener la lista de todos los archivos en la carpeta
archivos = os.listdir(carpeta)
imagenes = []
# Recorrer la lista de archivos y cargar las imágenes
for archivo in archivos:
    ruta_archivo = os.path.join(carpeta, archivo)
    if os.path.isfile(ruta_archivo) and archivo.endswith(('.jpg', '.jpeg', '.png')):
        # Cargar la imagen
        imagenes.append(cv2.imread(ruta_archivo))

for img in imagenes:
    points = mark_rectangle(img)
    all_points.append(points)

# Nombre del archivo CSV o TXT
output_filename = 'puntos.csv'

# Guardar los puntos en el archivo CSV o TXT
save_points_to_csv(all_points, output_filename)

print("Puntos guardados en el archivo:", output_filename)

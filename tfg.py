import cv2
import os
import seaborn as sns
import rawpy
import matplotlib.pyplot as plt
import csv
import pandas as pd
os.environ['OMP_NUM_THREADS'] = '10'
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


class Illuminant_estimation():

    def __init__(self):
        self.images = []
        self.corrected_images = []
        self.groundtruth_images = []
        self.data_images = []
        self.r_groundtruth = []
        self.g_groundtruth = []
        self.r_trees = []
        self.g_trees = []
        self.ruta_images = os.path.abspath("galaxy")
        self.raw_1illuminant_images = []
        self.raw_2illuminant_images = []
        self.load_images()

    def save_images(self):
        # Cargamos las imagenes
        for i in range(0, 50):
            nombre_carpeta = f"Place{i}"
            ruta = os.path.join(self.ruta_images, nombre_carpeta)
            for image in os.listdir(ruta):
                if image.endswith("_1.jpg"):
                    ruta_imagen = os.path.join(ruta, image)
                    imagen = cv2.imread(ruta_imagen)
                    self.images.append(imagen)
                elif image.endswith("_12.jpg"):
                    ruta_imagen = os.path.join(ruta, image)
                    imagen = cv2.imread(ruta_imagen)
                    self.groundtruth_images.append(imagen)
                elif image.endswith("_1.dng"):
                    ruta_imagen = os.path.join(ruta, image)
                    with rawpy.imread(ruta_imagen) as raw:
                        # Convertir el archivo RAW en una matriz de imagen RGB
                        imagen_rgb = raw.postprocess(output_bps=8)

                    # Convertir de RGB a BGR
                    bgr_image = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)

                    # Guardar la imagen en formato BGR
                    cv2.imwrite(f'raw_images_1illuminants/{str(i)}.jpg', bgr_image)
                elif image.endswith("_12.dng"):
                    ruta_imagen = os.path.join(ruta, image)
                    with rawpy.imread(ruta_imagen) as raw:
                        # Convertir el archivo RAW en una matriz de imagen RGB
                        imagen_rgb = raw.postprocess(output_bps=8)

                    # Convertir de RGB a BGR
                    bgr_image = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)

                    # Guardar la imagen en formato BGR
                    cv2.imwrite(f'raw_images_2illuminants/{str(i)}.jpg', bgr_image)

    def load_images(self):
        carpeta = "raw_images_2illuminants"
        # Obtener la lista de todos los archivos en la carpeta
        archivos = os.listdir(carpeta)

        # Recorrer la lista de archivos y cargar las imágenes
        for archivo in archivos:
            ruta_archivo = os.path.join(carpeta, archivo)
            if os.path.isfile(ruta_archivo) and archivo.endswith(('.jpg', '.jpeg', '.png')):
                # Cargar la imagen
                img = cv2.imread(ruta_archivo)
                self.raw_2illuminant_images.append(img)

    def load_data(self):
        # Ruta de la carpeta que contiene las imágenes TIFF
        data = 'data'
        # Obtener la lista de archivos en la carpeta
        archivos = os.listdir(data)
        # Filtrar solo los archivos TIFF (por extensión)
        archivos_tiff = [archivo for archivo in archivos if archivo.endswith('.tiff')]

        # Cargar todas las imágenes en una lista
        self.data_images = []
        for archivo_tiff in archivos_tiff:
            ruta_imagen = os.path.join(data, archivo_tiff)
            imagen = cv2.imread(ruta_imagen)
            self.data_images.append(imagen)

        # Cargar datos groundtruth
        datos = pd.read_csv('data.csv', header=None)
        RGB_values = datos.values
        groundtruth = [(r / (r + g + b), g / (r + g + b)) for r, g, b in RGB_values]
        self.r_groundtruth, self.g_groundtruth = zip(*groundtruth)

    def regression_tree_training(self, num_trees=5, num_features=4):
        self.load_data()
        features = [[], [], [], []]
        for image in self.data_images:
            feat = self.extract_features(image)
            for i in range(num_features):
                features[i].append(feat[i])

        self.r_trees = []
        self.g_trees = []
        for i in range(num_trees):
            r_list = []
            g_list = []
            for j in range(num_features):
                r_tree = DecisionTreeRegressor()
                g_tree = DecisionTreeRegressor()
                r_tree.fit(features[j], self.r_groundtruth)
                g_tree.fit(features[j], self.g_groundtruth)
                r_list.append(r_tree)
                g_list.append(g_tree)
            self.r_trees.append(r_list)
            self.g_trees.append(g_list)

    def regression_tree_method(self, img, num=None, num_features=4):
        if not self.r_trees or not self.g_trees:
            self.regression_tree_training(10, num_features)
        sub_images = self.divide_image(img, 80, 80)
        points_rg = []

        for sub in sub_images:
            feature = self.extract_features(sub)
            all_candidates = []

            for tree_idx, (tree_r, tree_g) in enumerate(zip(self.r_trees, self.g_trees)):
                candidates = []

                for j in range(num_features):
                    res_r = tree_r[j].predict(feature[j].reshape(1, -1))[0]
                    res_g = tree_g[j].predict(feature[j].reshape(1, -1))[0]
                    candidates.append((res_r, res_g))

                all_candidates.append(candidates)

            agreed_candidates = []

            for i in range(num_features):
                consensus_count = 0

                for j in range(num_features):
                    if i != j:
                        distance = np.linalg.norm(np.array(all_candidates[i]) - np.array(all_candidates[j]))
                        if distance <= 0.025:
                            consensus_count += 1

                if consensus_count >= 3:
                    agreed_candidates.extend(all_candidates[i])

            if len(agreed_candidates) > 0:
                points_rg.append(self.median(agreed_candidates))
            else:
                for candidates in all_candidates:
                    points_rg.extend(candidates)

        if not points_rg:
            print("Ningún árbol ha sido válido")
        else:
            self.result(img, points_rg, "RegressionTree", num)

    def divide_image(self, img, sub_height=20, sub_width=20):
        height, width, channels = img.shape
        sub_images = []
        for y in range(0, height, sub_height):
            for x in range(0, width, sub_width):
                sub_image = img[y:y + sub_height, x:x + sub_width, :]
                sub_images.append(sub_image)
        return sub_images

    def gray_world_method(self, img, num=None):
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.gray_world(sub))
            points_rg.append(self.calcule_rg(points[-1]))

        self.result(img, points_rg, "GrayWorld", num)

    def general_gray_world_method(self, img, num=None):
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.general_gray_world(sub))
            points_rg.append(self.calcule_rg(points[-1]))

        self.result(img, points_rg, "GeneralGrayWorld", num)

    def white_patch_method(self, img, num=None):
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.white_patch(sub))
            points_rg.append(self.calcule_rg(points[-1]))

        self.result(img, points_rg, "WhitePatch", num)

    def max_RGB_method(self, img, num=None):
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.max_RGB(sub))
            points_rg.append(self.calcule_rg(points[-1]))
        self.result(img, points_rg, "MaxRGB", num)

    def shades_of_gray_method(self, img, num=None):
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.shades_of_gray(sub))
            points_rg.append(self.calcule_rg(points[-1]))
        self.result(img, points_rg, "ShadesOfGray", num)

    def histogram_stretch_method(self):
        img = self.images[9]
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.histogram_stretch(sub))
            points_rg.append(self.calcule_rg(points[-1]))
        self.result(img, points_rg, "HistogramStretch")

    def corrected_moment_method(self, img, num=None):
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.corrected_moment(sub))
            points_rg.append(self.calcule_rg(points[-1]))
        self.result(img, points_rg, "CorrectedMoment", num, True)

    def result(self, img, points_rg, name, num=None, save=False):
        # Calculamos la average pairwise distance
        avg_distance = self.average_pairwise_distance(points_rg)
        print(f"Average pairwise distance: {avg_distance}")

        if avg_distance <= 0.025:
            print("Una sola iluminacion")
            L = self.median(points_rg)
            print(f"Iluminante: {L}")
            corrected_image = self.color_correction(img, L)
            cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"Resultados/{name}/{num}_ImagenCorregida_{name}1_iluminante.jpg", corrected_image)
        else:
            print("Mas de una iluminacion")
            centroids = self.get_kmeans_centroids(points_rg)
            L1 = centroids[0]
            L5 = centroids[1]

            # Calcula las diferencias absolutas entre R y G para cada iluminante
            diff_L1 = np.abs(L1[0] - L1[1])
            diff_L5 = np.abs(L5[0] - L5[1])

            # Compara las diferencias para determinar cuál iluminante tiene una menor diferencia entre R y G
            if diff_L1 > diff_L5:
                natural_light_rg = L1
                artificial_light_rg = L5
            else:
                natural_light_rg = L5
                artificial_light_rg = L1

            print(f"Iluminante 1: {natural_light_rg}, Iluminante 2: {artificial_light_rg}")
            L = {1.0: "L1", .75: "L2", .5: "L3", .25: "L4", 0: "L5"}
            alfa_values = [1.0, .75, .5, .25, 0]
            for alfa in alfa_values:
                illumination = natural_light_rg * alfa + artificial_light_rg * (1 - alfa)
                if save:
                    corrected_image = self.color_correction(img, illumination)
                    cv2.imwrite(f"Verificacion/{name}/ImagenCorregida_{name}{L[alfa]}.jpg", corrected_image)
            illumination = natural_light_rg * 0.25 + artificial_light_rg * (1 - 0.25)
            corrected_image = self.color_correction(img, illumination)
            cv2.imwrite(f"Resultados/{name}/{num}_ImagenCorregida_{name}L2.jpg", corrected_image)

    def calcule_rg(self, vector):
        R, G, B = vector
        r = R / (R + G + B)
        g = G / (R + G + B)
        return r, g

    def gray_world(self, img):
        # Convertir la imagen a punto flotante
        image_float = img.astype(np.float64) / 255.0

        # Dividir la imagen en los canales de color RGB
        B, G, R = cv2.split(image_float)

        # Calcular las medias de cada canal RGB
        avgB = np.mean(B)
        avgG = np.mean(G)
        avgR = np.mean(R)

        # Calcular el factor de corrección para cada canal
        grayValue = (avgB + avgG + avgR) / 3.0
        bCorr = grayValue / (avgB + 1e-10)  # Agregar un pequeño valor para evitar divisiones por cero
        gCorr = grayValue / (avgG + 1e-10)
        rCorr = grayValue / (avgR + 1e-10)

        return rCorr, gCorr, bCorr

    def general_gray_world(self, img):
        # Convertir la imagen a flotante y escalar los valores a [0, 1]
        img_float = img.astype(np.float32) / 255.0

        b, g, r = cv2.split(img_float)

        # Calcular la media de cada canal de color
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)

        # Calcular el factor de corrección para cada canal
        rCorr = (mean_g / mean_r)
        gCorr = 1.0
        bCorr = (mean_g / mean_b)

        return rCorr, gCorr, bCorr

    def white_patch(self, img):
        # Buscar los valores máximos en cada canal
        max_values = np.max(img, axis=(0, 1))

        # Calcular los factores de corrección
        rCorr = max_values[1] / max_values[0] if max_values[0] != 0 else 1.0
        gCorr = 1.0
        bCorr = max_values[1] / max_values[2] if max_values[2] != 0 else 1.0

        return rCorr, gCorr, bCorr

    def max_RGB(self, img):
        # Encontrar el valor máximo en cada canal de color
        max_r = np.max(img[:, :, 2])
        max_g = np.max(img[:, :, 1])
        max_b = np.max(img[:, :, 0])

        # Calcular los factores de corrección
        rCorr = max_r / np.mean(img[:, :, 2])
        gCorr = max_g / np.mean(img[:, :, 1])
        bCorr = max_b / np.mean(img[:, :, 0])

        return rCorr, gCorr, bCorr

    def shades_of_gray(self, img):
        # Calcular el canal promedio
        avg_gray = np.mean(img)

        # Calcular los factores de corrección
        rCorr = avg_gray / np.mean(img[:, :, 2])
        gCorr = avg_gray / np.mean(img[:, :, 1])
        bCorr = avg_gray / np.mean(img[:, :, 0])

        return rCorr, gCorr, bCorr

    def histogram_stretch(self, img):
        # Calcular los histogramas de los canales de color
        hist_r, _ = np.histogram(img[:, :, 2], bins=256, range=[0, 256])
        hist_g, _ = np.histogram(img[:, :, 1], bins=256, range=[0, 256])
        hist_b, _ = np.histogram(img[:, :, 0], bins=256, range=[0, 256])

        # Encontrar los valores más frecuentes en cada canal
        peak_r = np.argmax(hist_r)
        peak_g = np.argmax(hist_g)
        peak_b = np.argmax(hist_b)

        # Calcular los factores de corrección
        rCorr = peak_g / peak_r if peak_r != 0 else 1.0
        gCorr = 1.0
        bCorr = peak_g / peak_b if peak_b != 0 else 1.0

        return rCorr, gCorr, bCorr

    def corrected_moment(self, img):
        # Convertir la imagen a float para cálculos precisos
        img_float = img.astype(np.float32)

        # Calcular las medias de los canales de color en formato BGR
        means = np.mean(img_float, axis=(0, 1))

        # Calcular las desviaciones estándar de cada canal en formato BGR
        stds = np.std(img_float, axis=(0, 1))

        # Calcular los factores de corrección en función de las medias y desviaciones estándar
        rCorr = means[2] / means[1]
        gCorr = 1.0
        bCorr = means[2] / means[0]

        # Ajustar los factores de corrección en función de las desviaciones estándar
        if stds[1] != 0:
            rCorr *= (stds[2] / stds[1]) ** 0.5
        if stds[0] != 0:
            bCorr *= (stds[2] / stds[0]) ** 0.5

        return rCorr, gCorr, bCorr

    def average_pairwise_distance(self, points):
        # Calcular la matriz de distancias
        distances = pdist(points)

        # Calcular la distancia promedio
        average_distance = distances.mean()

        return average_distance

    def median(self, points):
        # Convertir la lista de puntos en un arreglo NumPy
        points_array = np.array(points)

        # Calcular la mediana de los puntos a lo largo del eje 0
        median = np.median(points_array, axis=0)

        return median

    def get_kmeans_centroids(self, points, k=2):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init="auto")
        kmeans.fit(points)
        centroids = kmeans.cluster_centers_
        return centroids

    def color_correction(self, img, illumination):
        img = img.astype(np.float32) / 255.0

        # Obtener los componentes de color R, G y B
        B, G, R = cv2.split(img)

        r = illumination[0]
        g = illumination[1]
        b = 1 - r - g

        """# Calcular la suma total de cada canal de color
        R_total = np.sum(R)
        G_total = np.sum(G)
        B_total = np.sum(B)

        # Calcular la corrección para cada canal
        R_corr = r * (R_total + G_total + B_total) / R_total
        G_corr = g * (R_total + G_total + B_total) / G_total
        B_corr = b * (R_total + G_total + B_total) / B_total
"""
        R_corr, G_corr, B_corr = self.solve_equation(r, g, b)
        # Aplicar la corrección a cada canal y asegurarse de que los valores estén en el rango [0, 1]
        newR = np.clip(R * R_corr, 0, 1)
        newG = np.clip(G * G_corr, 0, 1)
        newB = np.clip(B * B_corr, 0, 1)

        # Combinar los canales y convertir la imagen a tipo uint8 en el rango [0, 255]
        corrected_img = cv2.merge((newB, newG, newR))
        corrected_img = (corrected_img * 255).astype(np.uint8)

        return corrected_img

    def solve_equation(self, r, g, b):
        G = 1.0
        R = (r * G) / g
        B = (1 - r - g) * G / g
        return R, G, B

    def extract_features(self, img):
        """
            Esta funcion se utiliza para extraer las características de la imagen

        """
        # Convertimos la imagen de BGR a espacio de color RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(img_rgb)
        # Extraemos los canales R, G y B de la imagen
        img_float = img.astype(np.float32) / 255.0
        r_f, g_f, b_f = cv2.split(img_float)

        # Calcular las características de cromaticidad promedio
        avg_chromaticity = np.mean(img_rgb, axis=(0, 1)) / np.sum(img_rgb)

        # Calcular las características de cromaticidad del color más brillante
        brightest_chromaticity = np.max(img_rgb, axis=(0, 1)) / np.sum(img_rgb)

        # Calculamos la cromaticidad dominante
        # Calcular la media de los canales de color
        mean_red = np.mean(r_f)
        mean_green = np.mean(g_f)
        mean_blue = np.mean(b_f)
        # Calcular la cromaticidad dominante
        dominant_chromaticity = [mean_red, mean_green, mean_blue]

        # Calculamos la cromaticidad de la moda
        palette = np.column_stack((r.flatten(), g.flatten(), b.flatten()))
        color_counts = np.bincount(palette.argmax(axis=1))
        color_mode = np.array([np.argmax(color_counts) // 3, np.argmax(color_counts) % 3])
        chromaticity_mode = color_mode / 255

        features = [avg_chromaticity, brightest_chromaticity, np.array(dominant_chromaticity, dtype=np.float64),
                    chromaticity_mode]

        return features

    def guardar_resultados(self):

        for num, img in enumerate(self.raw_2illuminant_images):
            num = str(num)
            # self.regression_tree_method(img, num)
            self.gray_world_method(img, num)
            self.white_patch_method(img, num)
            self.corrected_moment_method(img, num)
            self.general_gray_world_method(img, num)
            self.max_RGB_method(img, num)
            self.shades_of_gray_method(img, num)

    def load_csv(self, filename):
        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Convertir la cadena de coordenadas a una tupla
                point1 = eval(row[0])
                data.append(point1)
        return data

    def crop_and_calculate_rgb(self, img, point):
        # Recortar el punto de la imagen
        x, y = point
        cropped_img = img[y:y + 1, x:x + 1]

        # Calcular los valores RGB
        rgb = cropped_img[0, 0]
        return rgb

    def euclidean_distance(self, color1, color2):
        color1 = np.array(color1)
        color2 = np.array(color2)
        distance = np.linalg.norm(color1 - color2)
        return distance

    def load_resultados(self, metodos=None):
        imagenes = {}
        for met in metodos:
            carpeta = "Resultados/" + met
            # Obtener la lista de todos los archivos en la carpeta
            archivos = os.listdir(carpeta)
            # Inicializar la lista de imágenes para la clave actual
            imagenes[met] = []
            self.resultados[met] = []
            # Recorrer la lista de archivos y cargar las imágenes
            for archivo in archivos:
                ruta_archivo = os.path.join(carpeta, archivo)
                if os.path.isfile(ruta_archivo) and archivo.endswith(('.jpg', '.jpeg', '.png')):
                    # Cargar la imagen
                    img = cv2.imread(ruta_archivo)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imagenes[met].append(img)
        return imagenes

    def comparar_resultados(self):
        colores = [(255, 0, 0),  # rojo
                   (0, 255, 0),  # verde
                   (0, 0, 255),  # azul
                   (255, 255, 0),  # amarillo
                   (150, 75, 0),  # marrón
                   (255, 165, 0),  # naranja
                   (255, 192, 203),  # rosa
                   (255, 255, 255),  # blanco
                   (128, 128, 128),  # gris
                   (0, 0, 0)]  # negro

        rojo = self.load_csv("puntos/rojo.csv")
        verde = self.load_csv("puntos/verde.csv")
        azul = self.load_csv("puntos/azul.csv")
        amarillo = self.load_csv("puntos/amarillo.csv")
        marron = self.load_csv("puntos/marron.csv")
        naranja = self.load_csv("puntos/orange.csv")
        rosa = self.load_csv("puntos/rosa.csv")
        blanco = self.load_csv("puntos/gris.csv")
        gris = self.load_csv("puntos/gris.csv")
        negro = self.load_csv("puntos/negro.csv")

        self.resultados = {}
        metodos = ["CorrectedMoment", "GeneralGrayWorld", "GrayWorld", "MaxRGB", "ShadesOfGray", "WhitePatch",
                   "RegressionTree"]
        imagenes = self.load_resultados(metodos)

        for num in range(len(rojo)):
            for met in metodos:
                i = imagenes[met][num]
                c = rojo[num]
                r = self.crop_and_calculate_rgb(imagenes[met][num], rojo[num])
                vector = [r, self.crop_and_calculate_rgb(imagenes[met][num], verde[num]),
                          self.crop_and_calculate_rgb(imagenes[met][num], azul[num]),
                          self.crop_and_calculate_rgb(imagenes[met][num], amarillo[num]),
                          self.crop_and_calculate_rgb(imagenes[met][num], marron[num]),
                          self.crop_and_calculate_rgb(imagenes[met][num], naranja[num]),
                          self.crop_and_calculate_rgb(imagenes[met][num], rosa[num]),
                          self.crop_and_calculate_rgb(imagenes[met][num], blanco[num]),
                          self.crop_and_calculate_rgb(imagenes[met][num], gris[num]),
                          self.crop_and_calculate_rgb(imagenes[met][num], negro[num])]
                self.resultados[met].append(vector)

        # Calcular la desviación estándar para cada vector
        distancia = {}
        desviaciones = {}

        for met in metodos:
            resultados_metodo = self.resultados[met]
            distancia[met] = [self.euclidean_distance(colores, vector) for vector in resultados_metodo]
            desviaciones[met] = [np.std(np.array(vector) - np.array(colores)) for vector in resultados_metodo]

        # Graficar los resultados
        media = {}
        mediana = {}
        min = {}
        max = {}
        rango = {}
        des = {}
        for met in metodos:
            media[met] = np.mean(distancia[met])
            mediana[met] = np.median(distancia[met])
            min[met] = np.min(distancia[met])
            max[met] = np.max(distancia[met])
            rango[met] = np.ptp(distancia[met])
            plt.plot(distancia[met], label=met)

        plt.xlabel('Imagen')
        plt.ylabel('Distancia Euclidiana')
        plt.legend()
        # Configurar estilo Seaborn
        sns.set(style='whitegrid')

        # Aplicar estilo Seaborn al gráfico
        sns.despine()
        # Guardar el gráfico en un archivo de imagen
        plt.savefig('Distancia.png')

        # Mostrar el gráfico
        plt.show()

        for met in metodos:
            media[met] = np.mean(desviaciones[met])
            mediana[met] = np.median(desviaciones[met])
            min[met] = np.min(desviaciones[met])
            max[met] = np.max(desviaciones[met])
            rango[met] = np.ptp(desviaciones[met])
            plt.plot(desviaciones[met], label=met)

        plt.xlabel('Imagen')
        plt.ylabel('Desviación Estándar')
        plt.legend()
        # Configurar estilo Seaborn
        sns.set(style='whitegrid')

        # Aplicar estilo Seaborn al gráfico
        sns.despine()
        # Guardar el gráfico en un archivo de imagen
        plt.savefig('Desviacion.png')

        # Mostrar el gráfico
        plt.show()
        # Convertir el diccionario a un DataFrame de Pandas
        df = pd.DataFrame.from_dict(distancia)

        # Nombre del archivo CSV
        output_filename = "resultados.csv"

        # Guardar el DataFrame en el archivo CSV
        df.to_csv(output_filename, index=False)
        # rojo, verde, azul, amarillo,marron, naranja, rosa, blanco, gris, negro

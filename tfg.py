import cv2
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import math
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sympy import symbols, Eq, solve
import pandas as pd
from sklearn.neighbors import KernelDensity

class Illuminant_estimation():

    def __init__(self):
        self.images = []
        self.corrected_images = []
        self.groundtruth_images = []
        self.data_images = []
        self.data_groundtruth = []
        self.regression_trees = []
        self.ruta_images = os.path.abspath("galaxy")
        self.load_images()

    def load_images(self):
        # Cargamos las imagenes
        for i in range(0, 10):
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

    def load_data(self):
        # Ruta de la carpeta que contiene las imágenes TIFF
        data = 'data_2_illuminants'
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
        datos = pd.read_csv('data_2_illuminant.csv', header=None)
        RGB_values = datos.values
        self.data_groundtruth = [(r / (r + g + b), g / (r + g + b)) for r, g, b in RGB_values]

    def regression_tree_training(self, num_trees=30):
        self.load_data()
        features = []
        for image in self.data_images:
            features.append(self.extract_features(image))

        self.regression_trees = []
        for i in range(num_trees):
            tree = RandomForestRegressor()
            tree.fit(features, self.data_groundtruth)
            self.regression_trees.append(tree)

    def regression_tree_method(self):
        if not self.regression_trees:
            self.regression_tree_training(1)
        img = self.images[9]
        sub_images = self.divide_image(img)
        points_rg = []
        for sub in sub_images:
            partial_res = []
            for tree in self.regression_trees:
                partial_res.append(tree.predict(self.extract_features(sub)))
            if np.var(partial_res) <= 0.0001:
                points_rg.append(np.median(partial_res))
        if not points_rg:
            print("Ningun arbol ha sido valido")
        else:
            self.result(img, points_rg, "RegressionTreePrimera_Version")



    def divide_image(self, img, sub_height=15, sub_width=20):
        height, width, channels = img.shape
        sub_images = []
        for y in range(0, height, sub_height):
            for x in range(0, width, sub_width):
                sub_image = img[y:y + sub_height, x:x + sub_width, :]
                sub_images.append(sub_image)
        return sub_images

    def gray_world_method(self):
        img = self.images[9]
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.gray_world(sub))
            points_rg.append(self.calcule_rg(points[-1]))

        self.result(img, points_rg, "GrayWorld")

    def general_gray_world_method(self):
        img = self.images[9]
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.general_gray_world(sub))
            points_rg.append(self.calcule_rg(points[-1]))

        self.result(img, points_rg, "GeneralGrayWorld")

    def white_patch_method(self):
        img = self.images[9]
        sub_images = self.divide_image(img)
        points = []
        points_rg = []
        for sub in sub_images:
            points.append(self.white_patch(sub))
            points_rg.append(self.calcule_rg(points[-1]))

        self.result(img, points_rg, "WhitePatch")

    def result(self, img, points_rg, name):
        # Calculamos la average pairwise distance
        avg_distance = self.average_pairwise_distance(points_rg)
        print(f"Average pairwise distance: {avg_distance}")

        if avg_distance <= 0.025:
            print("Una sola iluminacion")
            L = self.median(points_rg)
            print(f"Iluminante: {L}")
            corrected_image = self.color_correction(img, L)
            cv2.imwrite(f"Resultados/ImagenCorregida_{name}1_iluminante.jpg", corrected_image)
        else:
            print("Mas de una iluminacion")
            centroids = self.get_kmeans_centroids(points_rg)
            L1 = centroids[0]
            L5 = centroids[1]
            print(f"Iluminante 1: {L1}, Iluminante 2: {L5}")
            L = {1.0: "L1", .75: "L2", .5: "L3", .25: "L4", 0: "L5"}
            mixture_of_illumination = [1.0, .75, .5, .25, 0]
            for alfa in mixture_of_illumination:
                illumination = L1 * alfa + L5 * (1 - alfa)
                corrected_image = self.color_correction(img, illumination)
                cv2.imwrite(f"Resultados/ImagenCorregida_{name}{L[alfa]}.jpg", corrected_image)

    def calcule_rg(self, vector):
        R, G, B = vector
        r = R / (R + G + B)
        g = G / (R + G + B)
        return r, g

    def gray_world(self, img):

        # Convertir la imagen a punto flotante
        image_float = img.astype(np.float32)

        # Calculamos la media de cada canal RGB
        B, G, R = cv2.split(image_float)
        avgB = np.average(B)
        avgG = np.average(G)
        avgR = np.average(R)

        # Calculamos el factor de corrección de cada canal
        grayValue = (avgB + avgG + avgR) / 3.0
        bCorr = grayValue / avgB
        gCorr = grayValue / avgG
        rCorr = grayValue / avgR

        return rCorr, gCorr, bCorr

    def general_gray_world(self, img):
        b, g, r = cv2.split(img)

        # Calcular los promedios de los canales de color
        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)

        bCorr = avg_g / avg_b
        gCorr = avg_g / avg_g
        rCorr = avg_g / avg_r

        return rCorr, gCorr, bCorr

    def white_patch(self, img):
        # Dividir la imagen en los canales de color (B, G, R)
        b, g, r = cv2.split(img.astype(np.float32))

        # Encontrar los valores máximos de intensidad en cada canal
        max_r = np.max(r)
        max_g = np.max(g)
        max_b = np.max(b)

        # Calcular los factores de corrección
        rCorr = 255 / max_r
        gCorr = 255 / max_g
        bCorr = 255 / max_b

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
        kmeans = KMeans(n_clusters=k, n_init="auto")
        kmeans.fit(points)
        centroids = kmeans.cluster_centers_
        return centroids

    def color_correction(self, img, illumination):
        # img = img.astype(np.float32) / 255.0
        # Obtener los componentes de color R, G y B
        B, G, R = cv2.split(img)
        r = illumination[0]
        g = illumination[1]
        b = 1 - r - g
        # Calcular la suma total de cada canal de color
        R_total = np.sum(R)
        G_total = np.sum(G)
        B_total = np.sum(B)

        # Calcular la corrección para cada canal
        R_corr = r * (R_total + G_total + B_total) / R_total
        G_corr = g * (R_total + G_total + B_total) / G_total
        B_corr = b * (R_total + G_total + B_total) / B_total

        # Aplicar la corrección a cada canal
        newR = R * R_corr
        newG = G * G_corr
        newB = B * B_corr

        # Combinar los canales
        corrected_img = cv2.merge((newB, newG, newR))
        return corrected_img

    def solve_equation(self, r, g, b, Rtotal, Gtotal, Btotal):
        # Definir las variables
        R, G, B = symbols('R G B')

        # Definir las ecuaciones
        eq1 = Eq(r * (Rtotal + Gtotal + Btotal), R)
        eq2 = Eq(g * (Rtotal + Gtotal + Btotal), G)
        eq3 = Eq(b * (Rtotal + Gtotal + Btotal), B)

        # Resolver el sistema de ecuaciones
        sol = solve((eq1, eq2, eq3), (R, G, B))

        RCorr = sol[R]
        GCorr = sol[G]
        BCorr = sol[B]
        return RCorr, GCorr, BCorr

    def extract_features(self, img):
        """
            Esta funcion se utiliza para extraer las características de la imagen

        """
        # Convertimos la imagen de BGR a espacio de color RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extraemos los canales R, G y B de la imagen
        r, g, b = cv2.split(img_rgb)

        # Calcular las características de cromaticidad promedio
        avg_chromaticity = np.mean(img_rgb, axis=(0, 1)) / np.sum(img_rgb)

        # Calcular las características de cromaticidad del color más brillante
        brightest_chromaticity = np.max(img_rgb, axis=(0, 1)) / np.sum(img_rgb)

        # Calculamos la cromaticidad dominante
        flattened = np.concatenate((r.flatten(), g.flatten(), b.flatten()))
        hist, _ = np.histogramdd(flattened.reshape(-1, 3), bins=256, range=[(0, 255), (0, 255), (0, 255)])
        flat_hist = hist.reshape(-1)
        max_idx = np.argmax(flat_hist)
        dominant_chromaticity = np.array([max_idx // (256 * 256), (max_idx // 256) % 256]) / 255

        # Calculamos la cromaticidad de la moda
        chromaticity_mode = self.calculate_chromaticity_mode(img_rgb)

        features = np.concatenate((avg_chromaticity, brightest_chromaticity, dominant_chromaticity, chromaticity_mode))

        return features

    def calculate_chromaticity_mode(self, image_rgb):
        # Construir la paleta de colores
        num_bins = 300
        hist, _ = np.histogramdd(image_rgb.reshape(-1, 3), bins=num_bins, range=[(0, 255), (0, 255), (0, 255)])

        # Obtener los colores de la paleta con más de un umbral de píxeles
        threshold = 200
        palette_colors = np.argwhere(hist > threshold)
        color_palette = palette_colors.mean(axis=1)

        # Normalizar los colores de la paleta
        normalized_palette = color_palette / np.sum(color_palette, axis=1, keepdims=True)

        # Estimar la densidad de kernel de las cromaticidades
        kde = KernelDensity(kernel='gaussian').fit(normalized_palette)
        density_scores = kde.score_samples(normalized_palette)

        # Encontrar el punto con la mayor densidad estimada
        mode_index = np.argmax(density_scores)
        chromaticity_mode = normalized_palette[mode_index]

        return chromaticity_mode
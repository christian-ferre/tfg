from tfg import Illuminant_estimation
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os


test = Illuminant_estimation()

test.guardar_resultados()

test.comparar_resultados()

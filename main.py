from tfg import Illuminant_estimation
import cv2
import matplotlib.pyplot as plt
import pandas as pd

test = Illuminant_estimation()

# test.gray_world_method()

# test.white_patch_method()

# test.general_gray_world_method()

test.regression_tree_method()
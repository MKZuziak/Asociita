import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_image(img):
    img = np.asarray(img)
    plt.imshow(img)


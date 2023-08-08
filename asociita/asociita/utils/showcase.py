import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random


def show_image(img):
    img = np.asarray(img)
    plt.imshow(img)


def show_random(imgs_orig, imgs_transformed):
    fig, ax = plt.subplots(2, 5)
    for position in range(5):
        random_pos = random.randint(0, len(imgs_orig) - 1)
        original_img = np.asarray(imgs_orig[random_pos])
        transformed_img = np.asarray(imgs_transformed[random_pos])
        ax[0][position].imshow(original_img)
        ax[1][position].imshow(transformed_img)
    plt.show()


def save_random(imgs_orig, imgs_transformed, transformation: str, name: str):
    fig, ax = plt.subplots(2, 5)
    for position in range(5):
        random_pos = random.randint(0, len(imgs_orig) - 1)
        original_img = np.asarray(imgs_orig[random_pos])
        transformed_img = np.asarray(imgs_transformed[random_pos])
        ax[0][position].imshow(original_img)
        ax[1][position].imshow(transformed_img)
    title = transformation + '_' + name + '.pdf'
    plt.savefig(title)


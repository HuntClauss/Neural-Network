from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from os import listdir


# def img2array(image):
#     img = Image.open(image).convert('LA')
#     img = img.resize((28, 28), Image.ANTIALIAS)
#     flat_a = np.asarray(img).ravel()
#     print(flat_a)
#     return flat_a.tolist()
w, h = 28, 28


def img2array(image):
    img = Image.open(image).convert('LA')
    img = img.resize((w, h))
    pixels = list(img.getdata())
    result = [pixels[i][0]/1000 for i in range(len(pixels))]
    # print(result)
    return result


img2array('male.jpg')

# url = r'DATA\Animals\01-animal-welfare-legislation-nationalgeographic_1429498.jpg'
# arr(url)

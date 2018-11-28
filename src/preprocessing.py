import keras.backend as K
import numpy as np
from PIL import Image


def preprocess(image=None,
               path_to_image=None,
               width=512,
               height=512):
    if image is None:
        image = load_image(path_to_image)
    image = image.resize((width, height))
    image_array = np.asarray(image, dtype='float32')
    image_array = np.expand_dims(image_array, axis=0)
    return K.variable(image_array[:, :, :, ::-1])


def postprocess_and_save(image, height, width, save_path):
    image = image.reshape((height, width, 3))
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype('uint8')
    image = Image.fromarray(image)
    image.save(save_path, format='png')
    return


def load_image(path):
    return Image.open(path)


def adjust_color(image_array):
    image_array[:, :, :, 0] -= 103.939
    image_array[:, :, :, 1] -= 116.779
    image_array[:, :, :, 2] -= 123.68
    return image_array

if __name__ == '__main__':
    img = load_image('/Users/max_dojo/repos/keras-style-transfer/pics/headshot.jpg')
    print(np.array(img).shape)
    var = preprocess(image=img, width=256, height=256)
    #postprocess_and_save(np.asarray(img), width=256, height=256, save_path='/Users/max_dojo/Desktop/test.jpg')

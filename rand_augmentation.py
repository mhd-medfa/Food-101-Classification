# Reference: https://github.com/heartInsert/randaugment/blob/master/Rand_Augment.py
# Rand Augmentation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

#==============
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import requests
import random
#==============

class Rand_Augment(ImageDataGenerator):
    def __init__(self, Numbers=None, max_Magnitude=None, **kwargs):
        '''
        Custom image data generator.
        Behaves like ImageDataGenerator, but allows color augmentation.
        '''
        super().__init__(preprocessing_function=self.__call__, **kwargs)
        
        self.transforms = ['autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize',
                           'contrast', 'brightness', 'sharpness', 'shearX', 'shearY', 'translateX', 'translateY']
        if Numbers is None:
            self.Numbers = len(self.transforms) // 2
        else:
            self.Numbers = Numbers
        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        fillcolor = 128
        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation ,
            # it is no  need to obey  the value  in  autoaugment.py
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 0.2, 10),
            "translateY": np.linspace(0, 0.2, 10),
            "rotate": np.linspace(0, 360, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 231, 10),
            "contrast": np.linspace(0.0, 0.5, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.3, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,           
            "invert": [0] * 10
        }
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fill=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fill=fillcolor),
            "rotate": lambda img, magnitude: self.rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: img,
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def rand_augment(self):
        """Generate a set of distortions.
             Args:
             N: Number of augmentation transformations to apply sequentially. N  is len(transforms)/2  will be best
             M: Max_Magnitude for all the transformations. should be  <= self.max_Magnitude """

        M = np.random.randint(0, self.max_Magnitude, self.Numbers)

        sampled_ops = np.random.choice(self.transforms, self.Numbers)
        return [(op, Magnitude) for (op, Magnitude) in zip(sampled_ops, M)]

    def __call__(self, image):
        try:
            operations = self.rand_augment()
            for (op_name, M) in operations:
                operation = self.func[op_name]
                mag = self.ranges[op_name][M]
                image = operation(image, mag)
        except:
            image = tf.keras.preprocessing.image.array_to_img(image)
            operations = self.rand_augment()
            for (op_name, M) in operations:
                operation = self.func[op_name]
                mag = self.ranges[op_name][M]
                image = operation(image, mag)
        return image

    def rotate_with_fill(self, img, magnitude):
        #  I  don't know why  rotate  must change to RGBA , it is  copy  from Autoaugment - pytorch
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

    def test_single_operation(self, image, op_name, M=-1):
        '''
        :param image: image
        :param op_name: operation name in   self.transforms
        :param M: -1  stands  for the  max   Magnitude  in  there operation
        :return:
        '''
        operation = self.func[op_name]
        mag = self.ranges[op_name][M]
        image = operation(image, mag)
        return image

def plot_augmentation(datagen, data, n_rows=1, n_cols=5):
    n_images = n_rows * n_cols
    gen_flow = datagen.flow(data)

    aspect_ratio = data.shape[1] / data.shape[2]
    base_size = 2
    fig_size = (n_cols*base_size/aspect_ratio, n_rows*base_size)
    fig = plt.figure(figsize=fig_size)

    for image_index in range(n_images):
        image = next(gen_flow)
        plt.subplot(n_rows, n_cols, image_index+1)
        plt.axis('off')
        plt.imshow(image[0], vmin=0, vmax=255)
    fig.tight_layout(pad=0.0)

if __name__ == "__main__":
    # url = 'https://github.com/dufourpascal/stepupai/raw/master/tutorials/data_augmentation/image_town.jpg'
    # r = requests.get(url, allow_redirects=True)
    # open('image.jpg', 'wb').write(r.content)

    image = load_img('data/waffles.jpg')
    image = img_to_array(image).astype(int)
    data = np.expand_dims(image, 0)
    plt.axis('off')
    plt.imshow(data[0])
    plt.show()

    datagen = Rand_Augment(Numbers=4, max_Magnitude=10)
    datagen.fit(data)
    plot_augmentation(datagen, data, n_rows=2, n_cols=6)
    plt.show()
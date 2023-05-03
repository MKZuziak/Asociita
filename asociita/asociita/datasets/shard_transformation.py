from typing import Any
from torchvision.transforms import Compose, GaussianBlur, RandomRotation, ToTensor, ToPILImage, RandomPerspective
import torch.random
from asociita.utils.custom_transformations import AddGaussianNoise

# Custom pipeline and fucntion to blur the image.
blurer = Compose([
    GaussianBlur(kernel_size=(5, 9), sigma=(0.9, 50))
    ])
def blur_img(shard):
    shard['image'] = blurer(shard['image'])

# Custom pipeline and function to rotate the image.
rotater = Compose([
    RandomRotation(degrees=(0, 270))
    ])
def rotate_img(shard):
    shard['image'] = rotater(shard['image'])
    return shard

# Custom pipeline and function to noise the image
noiser = Compose([
    ToTensor(),
    AddGaussianNoise(150., 250.),
    ToPILImage()
])
def noise_img(shard):
    shard['image'] = noiser(shard['image'])
    return shard

# Custom pipeline and function to apply random perspective
perspective = Compose([
    RandomPerspective(distortion_scale=0.6, p=1.0)
])
def perspective_img(shard):
    shard['image'] = perspective(shard['image'])
    return shard


class Shard_Transformation:
    '''A common class for a set of transformation static methods 
    that can be applied to a shard.'''
    @staticmethod
    def transform(shard, preferences):
        if preferences == 'noise':
            return Shard_Transformation.noise(shard)            
        elif preferences == 'blur':
            return Shard_Transformation.blur(shard)
        elif preferences == 'rotation':
            return Shard_Transformation.rotate(shard)
        elif preferences == 'perspective_change':
            return Shard_Transformation.change_perspective(shard)
        else:
            print("Invalid key-word argument")
            return shard
    

    def noise(shard):
        shard = shard.map()
        return shard
    

    def blur(shard):
        shard = shard.map(blur_img)
        return shard
    

    def rotate(shard):
        shard = shard.map(rotate_img)
        return shard
    

    def change_perspective(shard):
        shard = shard.map(perspective_img)
        return shard
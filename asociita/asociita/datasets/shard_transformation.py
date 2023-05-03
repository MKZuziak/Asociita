from typing import Any
from torchvision.transforms import Compose, GaussianBlur, RandomRotation, ToTensor, ToPILImage, RandomPerspective
from asociita.utils.custom_transformations import AddGaussianNoise
from datasets import arrow_dataset

# Custom pipeline and fucntion to blur the image.
# Pipeline
blurer = Compose([
    GaussianBlur(kernel_size=(5, 9), sigma=(0.9, 50))
    ])
# Transforming function
def blur_img(shard: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
    shard['image'] = [blurer(image) for image in shard['image']]
    return shard

# Custom pipeline and function to rotate the image.
# Pipeline
rotater = Compose([
    RandomRotation(degrees=(0, 270))
    ])
# Transform function
def rotate_img(shard: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
    shard['image'] = [rotater(image) for image in shard['image']]
    return shard

# Custom pipeline and function to noise the image
# Pipeline
noiser = Compose([
    ToTensor(),
    AddGaussianNoise(150., 250.),
    ToPILImage()
])
# Transform function
def noise_img(shard: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
    shard['image'] = [noiser(image) for image in shard['image']]
    return shard

# Custom pipeline and function to apply random perspective
# Pipeline
perspective = Compose([
    RandomPerspective(distortion_scale=0.6, p=1.0)
])
# Transform function
def perspective_img(shard: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
    shard['image'] = [perspective(image) for image in shard['image']]
    return shard


class Shard_Transformation:
    '''A common class for a set of transformation static methods 
    that can be applied to a shard.'''
    @staticmethod
    def transform(shard: arrow_dataset.Dataset, preferences: str) -> arrow_dataset.Dataset:
        """Performes transformation of the provided shard according to the preferences.
        -------------
        Args
            shard (arrow_dataset.Dataset): shard to be transformed.
            preferences (str): type of transformation that should be applied.
       -------------
        Returns
            arrow_dataset.Dataset"""
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
    

    @staticmethod
    def noise(shard: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
        shard = shard.map(noise_img, batched=True)
        return shard
    

    @staticmethod
    def blur(shard: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
        shard = shard.map(blur_img, batched = True)
        return shard
    

    @staticmethod
    def rotate(shard: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
        shard = shard.map(rotate_img, batched = True)
        return shard
    

    @staticmethod
    def change_perspective(shard: arrow_dataset.Dataset) -> arrow_dataset.Dataset:
        shard = shard.map(perspective_img, batched = True)
        return shard
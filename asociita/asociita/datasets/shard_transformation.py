from typing import Any
from torchvision.transforms import Compose, GaussianBlur, RandomRotation, ToTensor, ToPILImage, RandomPerspective
from asociita.utils.custom_transformations import AddGaussianNoise
from datasets import arrow_dataset


def blur_img(shard: arrow_dataset.Dataset,
             kernel_size: tuple[int, int]=(1, 3),
             sigma: tuple[float, float]=(3., 40.)) -> arrow_dataset.Dataset:
    """Blurs the given dataset.
        -------------
        Args
            kernel_size (tuple[int, int]) - size of the Gaussian kernel
            sigma (tuple[float, float]) - SD to be used to creating kernel to perform blurring, uniformly at [min, max]
       -------------
         Returns
            arrow_dataset.Dataset"""
    blurer = Compose([
    GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    ])
    shard['image'] = [blurer(image) for image in shard['image']]
    return shard


def rotate_img(shard: arrow_dataset.Dataset,
               degrees: tuple[int, int] = (0, 45)) -> arrow_dataset.Dataset:
    """Rotates the given dataset.
    -------------
    Args
        degrees (tuple[int, int]): the range expressed in degress to which the image can be rotated.
    -------------
    Returns
        arrow_dataset.Dataset"""
    rotater = Compose([
    RandomRotation(degrees=degrees)
    ])
    shard['image'] = [rotater(image) for image in shard['image']]
    return shard


def noise_img(shard: arrow_dataset.Dataset,
              noise_multiplyer: float = 0.009) -> arrow_dataset.Dataset:
    """Add gausian noise to the dataset.
    -------------
    Args
        noise_multiplyer (float): Noise multiplication (higher rates implicates adding more noise)
    -------------
    Returns
        arrow_dataset.Dataset"""
    noiser = Compose([
    ToTensor(),
    AddGaussianNoise(mean = 0., 
                     std = 1.,
                     noise_multiplication = noise_multiplyer),
    ToPILImage()
    ])  
    shard['image'] = [noiser(image) for image in shard['image']]
    return shard


def perspective_img(shard: arrow_dataset.Dataset,
                    distortion_scale: float =  0.5,
                    transformation_probability: float = 0.5) -> arrow_dataset.Dataset:
    """Changes the perspective of the images in the dataset.
    -------------
    Args
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
        transformation_probability (float): probability of the image being transformed.
    -------------
    Returns
        arrow_dataset.Dataset"""
    perspective = Compose([
    RandomPerspective(distortion_scale=distortion_scale, p=transformation_probability)
    ])
    shard['image'] = [perspective(image) for image in shard['image']]
    return shard


class Shard_Transformation:
    '''A common class for a set of transformation static methods 
    that can be applied to a shard.'''
    @staticmethod
    def transform(shard: arrow_dataset.Dataset, preferences: dict) -> arrow_dataset.Dataset:
        """Performes transformation of the provided shard according to the preferences.
        -------------
        Args
            shard (arrow_dataset.Dataset): shard to be transformed.
            preferences (dict): dict containing all the informations regarding the transformation.
       -------------
        Returns
            arrow_dataset.Dataset"""
        transformation_name = preferences["transformation_type"]
        
        if transformation_name == 'noise':
            return Shard_Transformation.noise(shard, preferences)            
        elif transformation_name == 'blur':
            return Shard_Transformation.blur(shard, preferences)
        elif transformation_name == 'rotation':
            return Shard_Transformation.rotate(shard, preferences)
        elif transformation_name == 'perspective_change':
            return Shard_Transformation.change_perspective(shard, preferences)
        else:
            print("Invalid key-word argument")
            return shard
    

    @staticmethod
    def noise(shard: arrow_dataset.Dataset, preferences: dict) -> arrow_dataset.Dataset:
        if preferences.get('noise_multiplyer'):
            noise_mult = preferences['noise_multiplyer']
        else:
            noise_mult = 0.005
        shard = shard.map(noise_img, batched=True, fn_kwargs={'noise_multiplyer': noise_mult})
        return shard
    

    @staticmethod
    def blur(shard: arrow_dataset.Dataset, preferences: dict) -> arrow_dataset.Dataset:
        if preferences.get('kernel_size'):
            kernel_size = tuple(preferences['kernel_size'])
        else:
            kernel_size = (1, 3)
        if preferences.get('sigma'):
            sigma = tuple(preferences['sigma'])
        else:
            sigma = (3, 40)
        shard = shard.map(blur_img, batched = True, fn_kwargs = {'kernel_size': kernel_size, 'sigma': sigma})
        return shard
    

    @staticmethod
    def rotate(shard: arrow_dataset.Dataset, preferences: dict) -> arrow_dataset.Dataset:
        if preferences.get('rotation'):
            degrees = tuple(preferences['rotation'])
        else:
            degrees = (0, 45)
        shard = shard.map(rotate_img, batched = True, fn_kwargs = {'degrees': degrees})
        return shard
    

    @staticmethod
    def change_perspective(shard: arrow_dataset.Dataset, preferences: dict) -> arrow_dataset.Dataset:
        if preferences.get('distortion_scale'):
            distortion_scale = preferences['distortion_scale']
        else:
            distortion_scale = 0.5
        if preferences.get('transformation_probability'):
            transformation_probability = preferences['transformation_probability']
        else:
            transformation_probability = 0.5
        shard = shard.map(perspective_img, batched = True, fn_kwargs = {'distortion_scale': distortion_scale, 'transformation_probability': transformation_probability})
        return shard
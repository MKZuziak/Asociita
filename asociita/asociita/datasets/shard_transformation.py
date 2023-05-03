from torchvision.transforms import Compose, GaussianBlur, RandomRotation

blurer = Compose([
    GaussianBlur(kernel_size=(5, 9), sigma=(0.9, 50))
    ])
def blur_img(shard):
    shard['image'] = blurer(shard['image'])


rotater = Compose([
    RandomRotation(degrees=(0, 270))
    ])
def rotate_img(shard):
    shard['image'] = rotater(shard['image'])
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
    

    def noise(shard):
        ...
    

    def blur(shard):
        shard = shard.map(blur_img)
        return shard
    

    def rotate(shard):
        shard = shard.map(rotate_img)
        return shard
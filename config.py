from typing import Literal

def get_mobilevit_config(
        config_arch: Literal['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s'], 
        num_classes: int , 
        image_size: int | None = 224, 
        image_channels: int | None = 3,
        dropout: float | None = 0.5
):
    """
    Get config for MobileVit.

    Args:
        config_arch: architecture of the model. Must be `'mobilevit_xxs'`, `'mobilevit_xs'`or `'mobilevit_s'`.
        num_classes (int): number of the classes. 
        image_size (int): image_size = image_height = image_width. Default to `224`.
        image_channels (int): channels of the input image. Default to `3`.
        dropout (float): dropout percentage. Default to `0.5`.
    """
    if config_arch == 'mobilevit_xxs':
        config = {
            'channels' : [image_channels, 16, 24, 24, 24, 48, 64, 80],
            'dims' : [64, 80, 96],
            'mlp_dims' : [128, 160, 192],
            'num_classes' : num_classes,
            'image_size' : image_size,
            'expansion_factor' : 2,
            'patch_size' : 2,
            'dropout' : dropout
        }

    elif config_arch == 'mobilevit_xs':
        config = {
            'channels' : [image_channels, 32, 48, 48, 48, 64, 80, 96],
            'dims' : [96, 120, 144],
            'mlp_dims' : [192, 240, 288],
            'num_classes' : num_classes,
            'image_size' : image_size,
            'expansion_factor' : 4,
            'patch_size' : 2 ,
            'dropout' : dropout
        }

    elif config_arch == 'mobilevit_s':
        config = {
            'channels' : [image_channels, 32, 64, 64, 64, 96, 128, 160],
            'dims' : [144, 192, 240],
            'mlp_dims' : [288, 384, 480],
            'num_classes' : num_classes,
            'image_size' : image_size,
            'expansion_factor' : 4,
            'patch_size' : 2,
            'dropout' : dropout
        }

    return config

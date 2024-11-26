import subprocess, os, warnings, sys, random
import tensorflow as tf
import numpy as np

random.seed(2023)
np.random.seed(2023)
tf.random.set_seed(2023)


if __name__ == '__main__':

    results_folder = 'results'
    try:
        os.makedirs(results_folder)
    except OSError:
        warnings.warn(f'Folder {results_folder} already exits.')

    # for non-vit models
    config_archs = ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'vit']

    for config_arch in config_archs:
        command = [
            sys.executable, 'main.py', config_arch,
            '--results_filename', os.path.join(results_folder, 'results_vit') if config_arch == 'vit' \
                else os.path.join(results_folder, 'results_not_vit') 
        ]
        subprocess.run(command)

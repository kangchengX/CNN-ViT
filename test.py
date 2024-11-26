import subprocess, os, warnings, sys 


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
                else os.path.join(results_folder, 'results_not_vit'),
            '--num_epochs', '1'
        ]
        subprocess.run(command)

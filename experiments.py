import subprocess, os, warnings

if __name__ == '__main__':

    results_folder = 'results'
    try:
        os.makedirs(results_folder)
    except OSError:
        warnings.warn(f'Folder {results_folder} already exits.')

    # for non-vit models
    config_archs = ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'vgg16', 'vgg19', 'resnet50', 'resnet101']

    for config_arch in config_archs:
        command = [
            'python', 'main.py',
            '--config_arch', config_arch,
            '--results_filename', os.path.join(results_folder, 'results_not_vit'),
            '--num_epochs', '2',
            '--image_size', '32',
            '--shuffle_images'
        ]
        subprocess.run(command)

    learning_rates = [1e-2, 1e-3, 1e-4]
    vit_depth_heads_list = [(8,8), (12,12)]
    batch_sizes = [16,32]

    # for vit models
    for learning_rate in learning_rates:
        for vit_depth_heads in vit_depth_heads_list:
            for batch_size in batch_sizes:
                command = [
                    'python', 'main.py',
                    '--config_arch', 'vit',
                    '--results_filename', os.path.join(results_folder, 'results_vit'),
                    '--vit_depth', f'{vit_depth_heads[0]}',
                    '--vit_num_heads', f'{vit_depth_heads[1]}',
                    '--num_epochs', '2',
                    '--image_size', '32',
                    '--shuffle_images'
                ]
                subprocess.run(command)
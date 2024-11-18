from setuptools import setup, find_packages

with open('requirements.txt') as f: 
    requirements = f.read().splitlines()

with open('README.md', encoding='utf-8') as f: 
    long_description = f.read()

setup(
    name='cnn-vit',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'cnn-vit=main:main',
        ],
    },
    author='Kangcheng Xu',
    author_email='kangcheng.xu@outlook.com',
    description='A simple command line tool to train and test models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kangchengX/CNN-ViT/tree/clt',
    python_requires='==3.10.4',
)

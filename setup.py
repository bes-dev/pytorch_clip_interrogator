"""
Copyright 2021 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from setuptools import setup, find_packages

readme = open('README.md').read()

VERSION = '2023.02.19.17'

requirements = [
    'wheel',
    'torch',
    'numpy',
    'git+https://github.com/huggingface/transformers.git',
    'tokenizers',
    'accelerate',
    'datasets',
    'addict==2.4.0',
    'tqdm==4.64.1',
    'faiss-cpu==1.7.3',
    'Pillow'
]

setup(
    # Metadata
    name='pytorch_clip_interrogator',
    version=VERSION,
    author='Sergei Belousov aka BeS',
    author_email='sergei.o.belousov@gmail.com',
    description='Prompt engineering tool using BLIP 1/2 + CLIP Interrogate approach.',
    long_description=readme,
    long_description_content_type='text/markdown',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],


    # install .json configs
    package_data={
        "pytorch_clip_interrogator": ["data/*.txt"]
    },
    include_package_data=True
)

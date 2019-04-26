#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name = "spray",
    version = "0.1",
    packages=find_packages(),
    install_requires=[
        'ntorx',
        'numpy>=1.15.4',
        'Pillow>=5.3.0',
        'torch>=0.4.1.post2',
        'torchvision>=0.2.1',
        'Click>=7.0',
    ],
    entry_points={
        'console_scripts': [
            'spray = spray.cli:main'
        ]
    }
)

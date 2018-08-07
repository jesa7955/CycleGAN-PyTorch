# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='CycleGAN-PyTorch',
    version='0.1.0',
    description='Implementation of CycleGAN',
    long_description=readme,
    author='Tong Li',
    author_email='litong@logos.t.u-toyko.ac.jp',
    url='https://github.com/jesa7955/CycleGAN-PyTorch.git',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)


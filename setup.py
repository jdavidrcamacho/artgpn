#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='artgpn',
      version='0.5',
      description='Implementation of a ARTificial Gaussian Processes Network in python3x',
      author='João Camacho',
      author_email='joao.camacho@astro.up.pt',
      license='MIT',
      url='https://github.com/jdavidrcamacho/artgpn',
      packages=['artgpn'],
      install_requires=[
        'numpy',
        'scipy',
        'emcee',
        'dynesty'
      ],
     )

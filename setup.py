#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='gpyrn',
      version='0.5',
      description='Implementation of a Gaussian processes regression network',
      author='João Camacho',
      author_email='joao.camacho@astro.up.pt',
      license='MIT',
      url='https://github.com/jdavidrcamacho/gpyrn',
      packages=['gpyrn'],
      install_requires=[
        'numpy',
        'scipy',
        'emcee',
        'dynesty'
      ],
     )

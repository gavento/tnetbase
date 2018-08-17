#!/usr/bin/env python3

from setuptools import setup

setup(name='tnetbase',
      version='0.1',
      description='Tom\'s tensorflow net base and utils',
      author='Tomas Gavenciak',
      author_email='gavento@ucw.cz',
      license='MIT',
      packages=['tnetbase'],
      install_requires=[
          'tensorflow',
          'networkx',
          'numpy',
      ],
      )

#!/usr/bin/env python

from setuptools import find_packages, setup

install_requires = []

__version__ = '0.1.0'

d = setup(
    name='surgical_adventure',
    version=__version__,
    packages=find_packages(exclude=['tests*', 'docs*']),
    install_requires=install_requires,
    author='Paul Pak',
    maintainer='ppak@mgh.harvard.edu',
    keywords='laparoscopic,mae,robotic,vision,segmentation,tracking',
    classifiers=['Environment :: Console'],
    description="",
    long_description="",
    license='MIT',
    test_suite='nose.collector',
    tests_require=['nose'],
    scripts=[],
)

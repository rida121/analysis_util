# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='analysis_util',
    version='0.1.0',
    description='Analysis utility for data science by python',
    long_description=readme,
    author='Yuya Fukumasu',
    author_email='rida0121@gmail.com',
    url='https://github.com/rida121/analysis_util',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)


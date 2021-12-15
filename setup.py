from setuptools import setup, find_packages

setup(
    name='meshfracs',
    version='1.0',
    packages=find_packages(exclude=['tests*']),
    license='none',
    description='a module for assigning conductivity and solid fraction to continent sized meshes',
    install_requires=[],
    author='Kevin A Mendoza',
    author_email='kevinmendoza@icloud.com')
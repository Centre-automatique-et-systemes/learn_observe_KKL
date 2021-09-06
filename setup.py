from setuptools import setup

setup(
    name='lena',
    url='https://github.com/lukasbahr/lena',
    author='Lukas Bahr',
    packages=['lena.net', 'lena.datasets', 'lena.observer', 'lena.util'],
    install_requires=['numpy', 'torch', 'scipy', 'matplotlib', 'torchdiffeq', 'smt', 'pyyaml'],
    version='0.01',
    license='MIT',
    description='Implementation of lena',
)

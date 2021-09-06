from setuptools import setup

setup(
    name='lena',
    url='https://github.com/Centre-automatique-et-systemes/lena',
    author='Lukas Bahr',
    packages=['lena'],
    install_requires=['numpy', 'torch', 'scipy', 'matplotlib', 'torchdiffeq', 'smt'],
    version='0.01',
    license='MIT',
    description='Implementation of lena',
)

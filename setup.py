from setuptools import setup

# Set matplotlib version because of issues of the current version with colab. Waiting for fix.

setup(
    name='learn_observe_KKL',
    url='https://github.com/Centre-automatique-et-systemes/learn_observe_KKL.git',
    author='Lukas Bahr',
    packages=['learn_KKL'],
    install_requires=['numpy', 'torch', 'scipy', 'matplotlib==3.5.1',
                      'torchdiffeq', 'smt', 'jupyter', 'tensorboard',
                      'seaborn', 'pytorch-lightning', 'dill', 'functorch'],
    version='0.1.0',
    license='MIT',
    description='Implementation of the paper: "Towards gain tuning for '
                'numerical KKL observers"',
)

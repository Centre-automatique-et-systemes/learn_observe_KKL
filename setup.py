from setuptools import setup

setup(
    name='learn_observe_KKL',
    url='https://github.com/Centre-automatique-et-systemes/learn_observe_KKL.git',
    author='Lukas Bahr',
    packages=['learn_KKL'],
    install_requires=['numpy', 'torch', 'scipy', 'matplotlib', 'torchdiffeq',
                      'smt', 'jupyter', 'tensorboard', 'seaborn'],
    version='0.1.0',
    license='MIT',
    description='Implementation of the paper: "Learning to observe with KKL '
                'observers"',
)

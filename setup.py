from setuptools import setup, find_packages

setup(
    name='linear-ess',
    version='0.1',
    description=('An implementation of elliptical slice sampling for multivariate truncated normal distributions'),
    author='Kaiwen Wu',
    author_email='kaiwenwu@seas.upenn.edu',
    install_requires=[
        "botorch",
    ],
    packages=find_packages(),
)

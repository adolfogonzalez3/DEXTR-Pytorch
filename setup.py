from setuptools import setup, find_packages
import sys


if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='DEXTR_PyTorch',
      packages=[package for package in find_packages()
                if package.startswith('DEXTR_PyTorch')],
      install_requires=[
          'numpy',
          'matplotlib'
          'python-opencv2',
          'pillow',
      ],
      description='A repository for training and running DEXTR models.',
      author='Adolfo Gonzalez III, Osvaldo Castellanos, David Parra',
      url='',
      author_email='',
      keywords="",
      license="",
      )



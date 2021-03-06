from setuptools import setup

setup(name='dataloadercache',
      version='0.1',
      description='test dataloader performance',
      python_requires='>=3.6',
      #   url='http://github.com/chroneus/DataloaderCache',
      author='Sergei Chicherin',
      author_email='serenkiy@ieee.org',
      license='Apache',
      packages=['dataloadercache','example'],
      install_requires=['torch', 'tqdm'],
      zip_safe=False)

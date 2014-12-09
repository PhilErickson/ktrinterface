
import os
import re
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))

def find_version(*file_paths):
    with open(os.path.join(here, *file_paths)) as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


config = {
    'description': 'A high-level API to the Ziena KNITRO solver.',
    'author': 'Philip J. Erickson',
    'url': 'https://github.com/PhilErickson/ktrinterface',
    'download_url':
        'https://github.com/PhilErickson/ktrinterface/archive/master.zip',
    'author_email': 'erickson.philip.j@gmail.com',
    'version': find_version('ktrinterface', '__init__.py'),
    'install_requires': ['nose>=1.3.0', 'pandas>=0.13.1', 'numpy>=1.8.0'],
    'packages': ['ktrinterface'],
    'scripts': [],
    'name': 'ktrinterface',
    'license': 'Apache',
    'classifiers': ['Development Status :: 3 - Alpha',
                    'Intended Audience :: Developers',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: Apache Software License',
                    'Programming Language :: Python :: 2',
                    'Programming Language :: Python :: 2.7'],
    'keywords': 'KNITRO, Ziena, Optimization, API'
}

setup(**config)

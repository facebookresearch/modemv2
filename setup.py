# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.             #                                                                                   # This source code is licensed under the MIT license found in the                   # LICENSE file in the root directory of this source tree. 


import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This library is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('modemv2')

setup(
    name='modemv2',
    version='0.1',
    packages=find_packages(),
    package_data={"": extra_files},
    include_package_data=True,
    description='visual mbrl',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url='https://github.com/palanc/modem.git',
)

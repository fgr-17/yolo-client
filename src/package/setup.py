#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from setuptools import find_packages, setup
from package import __version__
import io
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with io.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="package",
    version=__version__,
    description="Basic template package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Federico Roux',
    author_email='rouxfederico@gmail.com',
    platforms=['any'],
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'examples']),
    # install_requires=['some_package']
)

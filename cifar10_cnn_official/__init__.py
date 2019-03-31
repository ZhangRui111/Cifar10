# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Makes helper libraries available in the cifar10 package.

For simple CPU/GPU project:
you need __init__.py, cifar10.py, cifar10_eval.py, cifar10_input.py, cifar10_train.py
Run cifar10_train.py to training the cnn.
Run cifar10_eval.py to evaluate the accuracy.
"""

from __future__ import absolute_import
# 加入绝对引入这个新特性，用import string来引入系统的标准string.py, 而用from pkg import string来引入当前目录下的string.py了
from __future__ import division
from __future__ import print_function

import cifar10
import cifar10_input

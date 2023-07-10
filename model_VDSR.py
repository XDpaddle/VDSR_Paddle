# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Realize the model definition function."""
from math import sqrt
'''
import torch
from torch import nn'''
import paddle
from paddle import nn
import paddle.nn.functional as F

class ConvReLU(nn.Layer):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2D(channels, channels, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.relu = nn.ReLU(True)

    def forward(self, x) -> paddle.Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out

'''
class VDSR(nn.Layer):
    def __init__(self) -> None:
        super(VDSR, self).__init__()
        # Input layer
        self.conv0= nn.Sequential(
            nn.Conv2D(1, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        '''
'''
        trunk = []
        for _ in range(18):
            trunk.append(ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)
        '''
'''
        self.conv1 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv15 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv17 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        self.conv18 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False),
            nn.ReLU(True),
        )
        # Output layer
        self.conv19 = nn.Conv2D(64, 1, (3, 3), (1, 1), (1, 1), bias_attr=False)



    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: paddle.Tensor) -> paddle.Tensor:
        identity = x

        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.conv13(out)
        out = self.conv14(out)
        out = self.conv15(out)
        out = self.conv16(out)
        out = self.conv17(out)
        out = self.conv18(out)
        out = self.conv19(out)
        out = paddle.add(out, identity)

        return out
'''
class VDSR(nn.Layer):
    def __init__(self) -> None:
        super(VDSR, self).__init__()
        # Input layer
        self.conv0= nn.Conv2D(1, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)

        self.conv1 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv2 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv3 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv4 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv5 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv6 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv7 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv8 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv9 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv10 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv11 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv12 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv13 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv14 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv15 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv16 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv17 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv18 = nn.Conv2D(64, 64, (3, 3), (1, 1), (1, 1), bias_attr=False)
        self.conv19 = nn.Conv2D(64, 1, (3, 3), (1, 1), (1, 1), bias_attr=False)
    def forward(self, x):
        identity = x

        x = self.conv0(x)
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.conv7(F.relu(x))
        x = self.conv8(F.relu(x))
        x = self.conv9(F.relu(x))
        x = self.conv10(F.relu(x))
        x = self.conv11(F.relu(x))
        x = self.conv12(F.relu(x))
        x = self.conv13(F.relu(x))
        x = self.conv14(F.relu(x))
        x = self.conv15(F.relu(x))
        x = self.conv16(F.relu(x))
        x = self.conv17(F.relu(x))
        x = self.conv18(F.relu(x))
        x = self.conv19(F.relu(x))

        out = paddle.add(x, identity)

        return out
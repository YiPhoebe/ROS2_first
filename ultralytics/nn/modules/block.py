from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn
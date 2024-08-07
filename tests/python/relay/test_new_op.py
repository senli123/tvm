import sys
from typing import Callable, Optional

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay, te
from tvm.error import TVMError
from tvm.relay import create_executor, transform
from tvm.relay.testing import check_grad, run_infer_type

from utils import ref_funcs

# executor_kind = tvm.testing.parameter("graph", "vm")
executor_kind = tvm.testing.parameter("graph")
# @tvm.testing.parametrize_targets("llvm", "cpu")  
# @tvm.testing.requires_llvm  
def test_unfold(executor_kind, target="llvm", dev=tvm.cpu(0)):
   
    import torch
    import torch.nn as nn
    xshape = (1,1,9,9)
    x = relay.var("x", relay.TensorType(xshape, "float32"))
    strides=(1, 1)
    padding=(0, 0)
    dilation=(1, 1)
    kernel_size=(3, 3)
    z = relay.nn.Unfold(x, strides, padding, dilation, kernel_size)
    func = relay.Function([x], z)
    x_array = torch.rand(xshape, dtype=torch.float)
    x_data = x_array.numpy()

    unfold = nn.Unfold(kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation)
    ref_res = unfold(x_array)

    op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
        x_data
    )
    nn.RReLU
    tvm.testing.assert_allclose(op_res.numpy(), ref_res.numpy(), rtol=1e-5)

import sys
from typing import Callable, Optional

import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
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
    kshape = (1,1,3,3)
    x = relay.var("x", relay.TensorType(xshape, "float32"))
    w = relay.var("w", shape=kshape, dtype='float32')
    strides=(1, 1)
    padding=(0, 0)
    dilation=(1, 1)
    kernel_size=(3, 3)
    z = relay.nn.unfold(data = x, weight = w, strides = strides, padding = padding, dilation = dilation,\
        groups = 1, channels=1, kernel_size=(3, 3))

    func = relay.Function([x, w], z)
    x_array = torch.rand(xshape, dtype=torch.float)
    x_data = x_array.numpy()
    w_data =  np.random.uniform(-1, 1, size=kshape).astype('float32')
    unfold = nn.Unfold(kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation)
    ref_res = unfold(x_array)
    # ref_res = tvm.topi.testing.conv2d_nchw_python(
    #         x_data.astype('float32'), w_data.astype('float32'), 1, padding, groups=1
    #     )
    op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
        x_data,w_data
    )
    tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

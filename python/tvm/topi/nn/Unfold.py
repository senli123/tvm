# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Unfold operators."""
from __future__ import absolute_import as _abs

import re
from collections import namedtuple
from typing import Optional, Sequence, Union

import numpy as np
import tvm
from tvm import auto_scheduler, te

from ..utils import get_const_int, get_const_tuple, simplify, tag
from .pad import pad
from .utils import get_pad_tuple, get_pad_tuple_generic
from .winograd_util import winograd_transform_matrices

def Unfold(
    inp: te.Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]],
    dilation: Union[int, Sequence[int]],
    kernel_size: Union[int, Sequence[int]],
    out_dtype: Union[str, None] = None,
    auto_scheduler_rewritten_layout: Optional[str] = "",
    meta_schedule_original_shape=None,
    auto_scheduler_should_rewrite_layout: bool = False,
):
    """Convolution operator in NCHW or NHWC layout.

    Supports 1D, 2D, 3D, ... and grouping.

    Parameters
    ----------
    inp : tvm.te.Tensor
        N-D with shape [batch, in_channel, in_height, in_width, ...] in `data_layout`

    stride : int or a list/tuple of dim ints
        (where dim=2 for NCHW, dim=1 for NCH, etc.)
        Stride size, or [stride_height, stride_width, ...]

    padding : int or a list/tuple of dim or 2*dim ints
        (where dim=2 for NCHW, dim=1 for NCH, etc.)
        padding size, or
        [pad_height, pad_width, ...] for dim ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 2*dim ints

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    data_layout : str
        Layout of the input. N indicates batch dimension, C indicates
        channels, any other character indicates HW (or H or HWD for 1D and 3D).

    kernel_layout: Optional[str]
        Layout of the filter. I indicates input channels, O indicates output channels,
        any other character indicates HW dimension of the filter (or H or HWD for 1D and 3D).
        If kernel_layout is empty, use data_layout to infer the default kernel_layout. Default
        kernel_layout is OIHW for NCHW data layout, HWIO for NHWC data layout.

    out_dtype : str
        Elements are converted to this type before elementwise multiplication
        and summation.

    auto_scheduler_rewritten_layout: str
        Layout from autoscheduler's layout rewritting.

    meta_schedule_original_shape : Optional[List[PrimExpr]]
        The original shape of the input tensor.

    auto_scheduler_should_rewrite_layout : bool
        Should auto scheduler be allowed to rewrite the layout of the filter
        tensor. Defaults to false. This can cause errors if used with grouped
        convs.

    Returns
    -------
    Output : tvm.te.Tensor
        N-D with shape [batch, out_channel, out_height, out_width, ...] in `data_layout`
    """
    # dim = len(inp.shape) - 2
    # if out_dtype is None:
    #     out_dtype = inp.dtype
    # assert isinstance(stride, int) or len(stride) == dim
    # assert isinstance(dilation, int) or len(dilation) == dim
    # if isinstance(stride, int):
    #     strides = [stride for _ in range(dim)]
    # else:
    #     strides = stride

    # if isinstance(dilation, int):
    #     dilations = [dilation for _ in range(dim)]
    # else:
    #     dilations = list(dilation)
    
    # if isinstance(kernel_size, int):
    #     kernel_sizes = [kernel_size for _ in range(dim)]
    # else:
    #     kernel_sizes = list(kernel_size)
    
    # if auto_scheduler_rewritten_layout:
    #     num_filter, _, *kernel_dimensions = auto_scheduler.get_shape_from_rewritten_layout(
    #         auto_scheduler_rewritten_layout,
    #         ["ff", "rc"] + [f"r{i}" for i in ["y", "x", "z"][: len(kernel_dimensions)]],
    #     )
       
    # batch, in_channel, *dimensions = np.array(get_const_tuple(inp.shape))
    # kernel_dimensions = np.array(kernel_sizes)
    # dilated_kernel_dimensions = [(k - 1) * dil + 1 for k, dil in zip(kernel_dimensions, dilations)]
    # pad_begin, pad_end = get_pad_tuple_generic(padding, dilated_kernel_dimensions)
   
    # # compute the output shape
    # out_channel = in_channel
    # out_dimensions = [
    #     simplify((d - (k - 1) * dil - 1 + pb + pe) // stride + 1)
    #     for d, k, dil, pb, pe, stride in zip(
    #         dimensions, kernel_dimensions, dilations, pad_begin, pad_end, strides
    #     )
    # ]
    # # compute graph
    # pad_before = list(np.array([0, 0] + pad_begin))
    # pad_after = list(np.array([0, 0] + pad_end))
    # temp = pad(inp, pad_before, pad_after, name="pad_temp")
    # rc = te.reduce_axis((0, in_channel), name="rc")
    # rs = [te.reduce_axis((0, k), name=f"r{i}") for i, k in zip(["y", "x"], kernel_dimensions)]

    # def compute(*args):
    #     nn, ff, *dim_indices = list(np.array(args))

    #     simplified_channel_index = rc
      
    #     return temp.__getitem__(
    #             tuple(
    #                 np.array(
    #                     [nn, simplified_channel_index]
    #                     + [
    #                         di * stride + r * dil
    #                         for di, stride, r, dil in zip(dim_indices, strides, rs, dilations)
    #                     ]
    #                 )
    #             )
    #         ).astype(out_dtype)

    # out = te.compute(
    #     list(np.array([batch, out_channel] + out_dimensions)),
    #     compute,
    #     name="Unfold.generic",
    #     tag="Unfold.generic",
    #     varargs_names=list(np.array(["nn", "ff", "yy", "xx"]))
    # )
    # return out
    
    ishape = inp.shape
    dim = 1
    for i in range(1, len(ishape)):
        dim = dim * ishape[i]
    oshape = [ishape[0], dim]
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    def unwrap(idx, shape):
        index = []
        for s in reversed(shape):
            index.append(idxmod(idx, s))
            idx = idxdiv(idx, s)
        return list(reversed(index))

    return te.compute(oshape, lambda i, j: inp(i, *unwrap(j, ishape[1:])),
                              name="Unfold.generic",
        tag="Unfold.generic",)

#     x = inp
#     return te.compute(x.shape, lambda *i: te.sigmoid(x(*i))
# , name="Unfold.generic",
#                     tag="Unfold.generic")
    # ishape = inp.shape
    # dim = 1
    # for i in range(1, len(ishape)):
    #     dim = dim * ishape[i]
    # oshape = [ishape[0], dim]
    # idxdiv = tvm.tir.indexdiv
    # idxmod = tvm.tir.indexmod

    # def unwrap(idx, shape):
    #     index = []
    #     for s in reversed(shape):
    #         index.append(idxmod(idx, s))
    #         idx = idxdiv(idx, s)
    #     return list(reversed(index))

    # return te.compute(oshape, lambda i, j: inp(i, *unwrap(j, ishape[1:])))
    
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
"""unfold operators."""
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

def unfold_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return unfold(Input, Filter, stride, padding, dilation, 1, "NCHW", "OIHW", out_dtype=out_dtype)



def unfold(
    inp: te.Tensor,
    filt: te.Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]],
    dilation: Union[int, Sequence[int]],
    groups: int,
    data_layout: str,
    kernel_layout: str = "",
    out_dtype: Union[str, None] = None,
    auto_scheduler_rewritten_layout: Optional[str] = None,
    meta_schedule_original_shape=None,
    auto_scheduler_should_rewrite_layout: bool = False,
):
    """Convolution operator in NCHW or NHWC layout.

    Supports 1D, 2D, 3D, ... and grouping.

    Parameters
    ----------
    inp : tvm.te.Tensor
        N-D with shape [batch, in_channel, in_height, in_width, ...] in `data_layout`

    filt : tvm.te.Tensor
        N-D with shape [num_filter, in_channel // groups, filter_height, filter_width, ...] in
        `kernel_layout`

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

    groups : int
        number of groups

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
    dim = len(inp.shape) - 2
    if out_dtype is None:
        out_dtype = inp.dtype
    assert isinstance(stride, int) or len(stride) == dim
    assert isinstance(dilation, int) or len(dilation) == dim
    if isinstance(stride, int):
        strides = [stride for _ in range(dim)]
    else:
        strides = stride

    if isinstance(dilation, int):
        dilations = [dilation for _ in range(dim)]
    else:
        dilations = list(dilation)

    # transform from data_layout to NCHW
    data_permutation_to = [data_layout.find("N"), data_layout.find("C")] + [
        x.span()[0] for x in re.finditer("[^NC]", data_layout)
    ]
    # transform from NCHW to data_layout
    data_permutation_from = np.argsort(data_permutation_to)
    # transform from CHW to data_layout
    data_permutation_from_reductions = data_permutation_from[1:].copy()
    data_permutation_from_reductions[
        data_permutation_from_reductions > data_permutation_from[0]
    ] -= 1

    if kernel_layout == "":
        # kernel permutation, if C appears before HW then num_filter is first, otherwise it is last
        # tkonolige: I don't really understand kernel ordering for NHWC, it seems
        # like num_filters should match the N dimension
        if data_layout.find("C") < re.search("[^NC]", data_layout).span()[0]:
            kernel_permutation_to = [0, 1] + list(range(2, dim + 2))
        else:
            kernel_permutation_to = [dim + 1, dim] + list(range(dim))
    else:
        # transform from kernel_layout to OIHW
        kernel_permutation_to = [kernel_layout.find("O"), kernel_layout.find("I")] + [
            x.span()[0] for x in re.finditer("[^OI]", kernel_layout)
        ]
    # transform from OIHW to kernel_layout
    kernel_permutation_from = np.argsort(kernel_permutation_to)

    if meta_schedule_original_shape:
        auto_scheduler.rewrite_tensor_shape(filt, meta_schedule_original_shape)
    batch, in_channel, *dimensions = np.array(get_const_tuple(inp.shape))[
        data_permutation_to
    ].tolist()
    num_filter, _, *kernel_dimensions = np.array(get_const_tuple(filt.shape))[
        kernel_permutation_to
    ].tolist()

    # Autoscheduler may have messed with the input layout, so we extract the
    # dimensions that it gives us
    if auto_scheduler_rewritten_layout:
        num_filter, _, *kernel_dimensions = auto_scheduler.get_shape_from_rewritten_layout(
            auto_scheduler_rewritten_layout,
            ["ff", "rc"] + [f"r{i}" for i in ["y", "x", "z"][: len(kernel_dimensions)]],
        )
        auto_scheduler.remove_index_check(filt)

    assert in_channel % groups == 0, "input channels must divide group size"
    assert num_filter % groups == 0, "output channels must divide group size"

    dilated_kernel_dimensions = [(k - 1) * dil + 1 for k, dil in zip(kernel_dimensions, dilations)]
    pad_begin, pad_end = get_pad_tuple_generic(padding, dilated_kernel_dimensions)
    # compute the output shape
    out_channel = num_filter
    out_dimensions = [
        simplify((d - (k - 1) * dil - 1 + pb + pe) // stride + 1)
        for d, k, dil, pb, pe, stride in zip(
            dimensions, kernel_dimensions, dilations, pad_begin, pad_end, strides
        )
    ]
    # compute graph
    pad_before = list(np.array([0, 0] + pad_begin)[data_permutation_from])
    pad_after = list(np.array([0, 0] + pad_end)[data_permutation_from])
    temp = pad(inp, pad_before, pad_after, name="pad_temp")
    rc = te.reduce_axis((0, in_channel // groups), name="rc")
    rs = [te.reduce_axis((0, k), name=f"r{i}") for i, k in zip(["y", "x", "z"], kernel_dimensions)]

    def compute(*args):
        nn, ff, *dim_indices = list(np.array(args)[data_permutation_to])

        if groups == 1:
            simplified_channel_index = rc
        else:
            simplified_channel_index = ff // (num_filter // groups) * (in_channel // groups) + rc

        return te.sum(
            temp.__getitem__(
                tuple(
                    np.array(
                        [nn, simplified_channel_index]
                        + [
                            di * stride + r * dil
                            for di, stride, r, dil in zip(dim_indices, strides, rs, dilations)
                        ]
                    )[data_permutation_from]
                )
            ).astype(out_dtype),
            # Schedules depend on reduction axes being in the same order as the
            # layout, so we reorder here.
            axis=np.array([rc, *rs])[data_permutation_from_reductions].tolist(),
        )
        
        # return temp.__getitem__(
        #         tuple(
        #             np.array(
        #                 [nn, simplified_channel_index]
        #                 + [
        #                     di * stride + r * dil
        #                     for di, stride, r, dil in zip(dim_indices, strides, rs, dilations)
        #                 ]
        #             )[data_permutation_from]
        #         )
        #     ).astype(out_dtype)

    out = te.compute(
        list(np.array([batch, out_channel] + out_dimensions)[data_permutation_from]),
        compute,
        # tag is expected to be lowercase
        tag=f"{'group_' if groups > 1 else ''}conv{dim}d_{data_layout.lower()}",
        name=f"{'group_' if groups > 1 else ''}conv{dim}d_{data_layout.lower()}",
        attrs={"layout_free_placeholders": [filt]} if auto_scheduler_should_rewrite_layout else {},
        varargs_names=list(np.array(["nn", "ff", "yy", "xx", "zz"])[data_permutation_from]),
    )
    # if we used autoscheduler's changed layout we need to rewrite the ordering
    # of the output dimensions
    if auto_scheduler_rewritten_layout:
        out = auto_scheduler.rewrite_compute_body(out, auto_scheduler_rewritten_layout)
    return out
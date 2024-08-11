import numpy as np
import torch
import torch.nn as nn
import tvm
import tvm.testing
from tvm import te

# def test_unfold():
#     dev = tvm.cpu(0)
#     tgt = tvm.target.Target(target="llvm", host="llvm")

#     b = te.var("b")
#     c = te.var("c")
#     h = te.var("h")
#     w = te.var("w")
#     c1 = te.var("c1")
#     A0 = te.placeholder((b, c, h, w), name="A0")
#     T0 = te.compute((b, c1, w), lambda i, j, k: A0[i, j//4, j%4, k], name="T")
#     s = te.create_schedule(T0.op)
#     unfold = tvm.build(s, [A0, T0], tgt, name="unfold")
#     b = 1
#     c = 3
#     h = 4
#     w = 4
#     c1 = 12
#     a = tvm.nd.array(np.random.uniform(size=(b,c,h,w)).astype(A0.dtype), dev)
#     c = tvm.nd.array(np.zeros(shape = (b,c1,w), dtype=T0.dtype), dev)
#     unfold(a, c)
#     a1 = a.numpy()
#     a1 = a1.reshape(b,c1,w)
#     tvm.testing.assert_allclose(c.numpy(), a1)

    # dev = tvm.cpu(0)
    # tgt = tvm.target.Target(target="llvm", host="llvm")
    # n = te.var("n")
    # A = te.placeholder((n,), name="A")
    # B = te.placeholder((n,), name="B")
    # C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    # s = te.create_schedule(C.op)
    # fadd = tvm.build(s, [A, B, C], tgt, name="myadd")
    # n = 1024
    # a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    # b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    # c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    # fadd(a, b, c)
    # tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


def test_unfold():
    dev = tvm.cpu(0)
    tgt = tvm.target.Target(target="llvm", host="llvm")
    
    def verify_unfold(input_shape, k, p, d, s):
        # init input shape
        n = input_shape[0]
        c = input_shape[1]
        h = input_shape[2]
        w = input_shape[3]
        # up_num = n*c*h*w
        A0 = te.placeholder((n, c, h, w), name="A0")
        # init unfold params and cal output shape
        on = n
        oc1 = k[0] * k[1]
        oh = (h + 2 * p[0] - (k[0] + 2 *(d[0] - 1))) // s[0] + 1
        ow = (w + 2 * p[1] - (k[1] + 2 *(d[1] - 1))) // s[1] + 1
        ol = oh * ow
        oc = c * oc1
        Apad = te.compute(
        (n, c, h + p[0]*2, w + p[1]*2),
        lambda nn, cc, yy, xx: tvm.tir.if_then_else(
            tvm.tir.all(yy >= p[0], yy - p[0] < h, xx >= p[1], xx - p[1] < w),
            A0[nn, cc, yy - p[0], xx - p[1]],
            0.0,
        ),
        name="Apad",
    )
        T0 = te.compute((on, oc, ol), 
                        lambda i, j, m: Apad[i, j // oc1,  
                                        m // ow * s[0] + j % oc1 // k[1] * d[0],
                                        m % ow * s[1] + j % oc1 % k[1] * d[1]], name="T")
        schedule = te.create_schedule(T0.op)
        unfold = tvm.build(schedule, [A0, T0], tgt, name="unfold")
    
        a = tvm.nd.array(np.random.uniform(size=(n, c, h, w)).astype(A0.dtype), dev)
        output = tvm.nd.array(np.zeros(shape = (on, oc, ol), dtype=T0.dtype), dev)
        unfold(a, output)
        a1 = torch.from_numpy(a.numpy())
        unfold = nn.Unfold(kernel_size = k,
            dilation = d,
            padding = p,
            stride = s)  
        unfolded = unfold(a1)
        tvm.testing.assert_allclose(output.numpy(), unfolded.numpy())

    verify_unfold(input_shape = (1,3,3,3), k = (2,2), p = (1,1), d = (1,1), s = (1,1))
    # verify_unfold(input_shape = (1,1,9,9), k = (3,3), p = (0,0), d = (3,3), s = (2,2))
    # verify_unfold(input_shape = (1,3,9,9), k = (3,3), p = (0,0), d = (3,3), s = (2,2))

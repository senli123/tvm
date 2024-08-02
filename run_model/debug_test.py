import numpy as np
import tvm
from tvm import  tir

rlt = tir.abs(-100)
print("abs(-100) = %d" % rlt)
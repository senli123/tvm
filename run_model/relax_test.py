import torch
import torch.nn as nn
import torch._dynamo as dynamo
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R, tir as T
import numpy as np
import tvm.testing
from pipeline import simlarity

def _convert_data_type(input_type):
    """converts the PyTorch scalar type input_type to a TVM dtype."""
    import torch  # type: ignore

    input_type = input_type.lower() if isinstance(input_type, str) else input_type
    if input_type == "float32":
        return torch.float32
    elif input_type == "float16":
        return torch.float16
    elif input_type == "int64":
        return torch.int64
    elif input_type == "int32":
        return torch.int32
    elif input_type == "bool":
        return torch.bool
    else:
        raise NotImplementedError("input_type {} is not handled yet".format(input_type))


def export_dynamo_model(model, input_datas):
    graph_model = dynamo.export(model, *input_datas)[0]
    return graph_model

def use_relax(input_info, graph_model, input_datas):
    input_datas = [ tvm.nd.array(input_data.numpy()) for input_data in input_datas]
    mod = from_fx(graph_model, input_info, unwrap_unit_return_tuple=True)
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    z = vm["main"](*input_datas)
    return z

def use_pytorch(model, args):
    pt_model = model
    return pt_model(*args)

class Sum(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x, y):
        return x + y
    
    
def run_dynamo_case():
    input_info = [([3, 4], "float32"), ([3, 4], "float32")]
    args = []
    for info in input_info:
        args.append(torch.randn(*info[0], dtype=_convert_data_type(info[1])))
    # export dynamo model
    graph_model = export_dynamo_model(Sum(), args)
    # use_relax
    relax_out = use_relax(input_info, graph_model, args)
    relax_out = [relax_o.numpy() for relax_o in relax_out] if isinstance(relax_out, list) else [relax_out.numpy()]
    # use pytorch
    pt_out = use_pytorch(Sum(), args)
    pt_out = [pt_o.numpy() for pt_o in pt_out] if isinstance(pt_out, list) else [pt_out.numpy()]
    simlarity(pt_out, relax_out)

if __name__ == "__main__":
    #run_dynamo_case（）
    saved_exported_program = torch.export.load(r'Y:\trans_onnx\miccai2022\GFNet-main\GFNet-main\exported_program_2.pt2')
    input_info = [([1, 450, 1000], "float32")]
    # print(saved_exported_program.graph)
    # mod = relay.frontend.from_pytorch(saved_exported_program, input_info)
    mod = from_fx(saved_exported_program.module(), input_info)



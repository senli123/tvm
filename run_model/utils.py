import os
import sys
import numpy as np
import torch
import torchvision.models as models
import onnx
import torch.nn as nn
import torch.nn.functional as F
from onnxsim import simplify
import platform
if platform.system() == "Windows":  
    save_dir = r'D:\project\programs\other_project\tvm_project\tvm\run_model\mode_zoo'
elif platform.system() == "Linux": 
    save_dir = '/workspace/trans_onnx/project/tvm/run_model/model_zoo'
else:    
    assert False, "noly support win and linux"


def trans_func(model, ir_path, model_name, pt_flag, args):
    if pt_flag:
        mod = torch.jit.trace(model, args)
        mod.save('{}.pt'.format(os.path.join(ir_path, model_name)))
    else:
        
        torch.onnx.export(model, args, os.path.join(ir_path, model_name +'.onnx'), verbose=True, input_names=['input0'],
                                output_names=['output0'],opset_version=11)
        model = onnx.load(os.path.join(ir_path, model_name +'.onnx'))
        # convert model
        model_simp, check = simplify(model)

        onnx.save(model_simp, '{}_sim.onnx'.format(os.path.join(ir_path, model_name)))
        






# model = models.resnet18(pretrained = True)
# model.eval()
# input = torch.randn([1,3,224,224])
# model_name = 'resnet18'


class TestModel(nn.Module):
    def __init__(self,):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(3,16,3)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class IndexModel(nn.Module):
    def __init__(self,):
        super(IndexModel, self).__init__()
       
    def forward(self,x):
        # x = x[:,:,[[1,2],[3,4]],:]
        x = x[:,:,:,:,[[1,2],[3,4]]]
        return x
    
class PermuteModel(nn.Module):
    def __init__(self,):
        super(PermuteModel, self).__init__()
       
    def forward(self,x):
        # x = x[:,:,[[1,2],[3,4]],:]
        x = x.permute(0,1,2,4,3,5)
        return x
    
    
class ReshapeModel(nn.Module):
    def __init__(self,):
        super(ReshapeModel, self).__init__()
       
    def forward(self,x):
        # x = x[:,:,[[1,2],[3,4]],:]
        x = x.reshape(1,147,3136)
        return x
    
class PadModel(nn.Module):
    def __init__(self,):
        super(PadModel, self).__init__()
       
    def forward(self,x):
        x = F.pad(x,[2,2,2,2])
        return x
    
class CatModel(nn.Module):
    def __init__(self,):
        super(CatModel, self).__init__()
        self.p = torch.rand([1,1,384])
    def forward(self,x):
        x = torch.cat([x, self.p],dim=1)
        return x


def export_model():
    pt_flag = False
    # model = TestModel()
    # model = IndexModel()
    # model = PermuteModel()
    # model = ReshapeModel()
    # model = PadModel()
    model = CatModel()
    model.eval()
    # input = torch.randn([1,3,224,224])
    # input = torch.randn([1,3,228,228])
    # input = torch.randn([1,3,7,56,228])
    # input = torch.randn([1,3,7,56,7,56])
    # input = torch.randn([1,3,7,7,56,56])
    # input = torch.randn([1,3,224,224])
    input = torch.randn([1,196,384])
    model_name = 'test_model'
    model_name = 'index'
    model_name = 'permute'
    model_name = 'reshape'
    model_name = 'pad'
    model_name = 'cat'
    if pt_flag:
        ir_path = os.path.join(save_dir, 'pt', model_name)
    else:
        ir_path = os.path.join(save_dir, 'onnx', model_name)

    os.makedirs(ir_path, exist_ok= True)
    # save model
    trans_func(model, ir_path, model_name, pt_flag, input)
    # save input
    input_array = input.numpy()
    input_array.tofile(os.path.join(ir_path, '{}_input.bin'.format(model_name)))
    
    
def gen_input_data():
    ir_path = r'D:\project\programs\other_project\tvm_project\tvm\run_model\mode_zoo\pt\index'
    input_shape = [4,2,3,4]
    input = torch.randn(input_shape)
    input_array = input.numpy()
    input_name = 'x'.join(str(s) for s in input_shape)
    input_array.tofile(os.path.join(ir_path, 'input0_{}.bin'.format(input_name)))
    
# gen_input_data()
export_model()
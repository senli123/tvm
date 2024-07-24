import os
import numpy as np
import torch
import torchvision.models as models
import onnx
from onnxsim import simplify
save_dir = '/workspace/trans_onnx/project/tvm/run_model/model_zoo'

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
        


pt_flag = False


model = models.resnet18(pretrained = True)
model.eval()
input = torch.randn([1,3,224,224])
model_name = 'resnet18'




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
import os
import numpy as np
import torch
import torchvision.models as models
save_dir = '/workspace/trans_onnx/project/tvm/run_model/model_zoo'
model = models.resnet18(pretrained = True)
model.eval()
input = torch.randn([1,3,224,224])
net = torch.jit.trace(model, input)
# save model
net.save(os.path.join(save_dir, 'resnet18.pt'))
# save input
input_array = input.numpy()
input_array.tofile(os.path.join(save_dir, 'resnet18_input.bin'))
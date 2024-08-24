
import os
import torch
import tvm
from tvm import relay

model_path = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/test_model.pt'
save_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/output3/output'
shape_list = [('x', [1,3,224,224])]
dump = False
load_model = True
if load_model:
    save_model_path = os.path.join(save_dir, 'model.json')
    save_param_path = os.path.join(save_dir, 'params.params')
    
    with open(save_model_path, 'r') as fi:  
        json_str = fi.read()  
        mod = tvm.ir.load_json(json_str)

    with open(save_param_path, "rb") as fo:
        param_str = fo.read()
        loaded_params = tvm.runtime.load_param_dict(param_str)
        
else:
    
    model = torch.load(model_path)
    mod, params, dump_tensor_names, dump_node_name_dict = relay.frontend.from_pytorch(model, shape_list, dump = dump)
    
    
print(123)

import os
import torch
import tvm
from tvm import relay

def get_relay(model_path, save_dir, shape_list, dump,  load_model):
    if load_model:
        save_model_path = os.path.join(save_dir, 'model.json')
        save_param_path = os.path.join(save_dir, 'params.params')
        
        with open(save_model_path, 'r') as fi:  
            json_str = fi.read()  
            mod = tvm.ir.load_json(json_str)

        with open(save_param_path, "rb") as fo:
            param_str = fo.read()
            params = tvm.runtime.load_param_dict(param_str)
            
    else:
        
        model = torch.load(model_path)
        mod, params, dump_tensor_names, dump_node_name_dict = relay.frontend.from_pytorch(model, shape_list, dump = dump)
    
    return mod, params


  
if __name__ == "__main__":
    # ----------------get relay-----------
    # sample 1
    model_path = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/test_model.pt'
    save_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/output3/output'
    shape_list = [('x', [1,3,224,224])]
    dump = False
    load_model = True 
    
    # sample 2
    
    model_path = '/workspace/my_tvm/model_zoo/pt/test_model/test_model.pt'
    save_dir = '/workspace/my_tvm/model_zoo/pt/test_model'
    shape_list = [('x', [1,3,224,224])]
    dump = False
    load_model = False 
    
    mod, params = get_relay(model_path, save_dir, shape_list, dump,  load_model)
    print(123)
    

from pipeline import export_pt_tvm
import os
import torch
# scnet50_torch21
# input_pt_path= 'Y:/trans_onnx/pnnx_test/scnet50_torch21/scnet50_torch21.pt'
# input_bin_path= 'Y:/trans_onnx/pnnx_test/scnet50_torch21/data_1x224x224x3.bin'
# input_shape = [1,3, 224, 224]
# target = "llvm"
# save_model = True
# save_dir = r'D:/project/programs/other_project/tvm_project/tvm_test_model/scnet50_torch21_2'


#faster rcnn
# input_pt_path= 'D:/project/model_zoo/faster-rcnn/end2end.onnx'
# input_bin_path= 'D:/project/model_zoo/faster-rcnn/input_0_1x3x800x1216.bin'
# input_shape = [1,3, 800, 1216]
# target = "llvm"
# save_model = True
# save_dir = r'D:/project/programs/other_project/tvm_project/tvm_test_model/faster_rcnn'

# os.makedirs(save_dir, exist_ok= True)
# load_model_flag = False
# export_pt_tvm(input_pt_path, input_bin_path, input_shape, target, save_model, load_model_flag, save_dir)


#t2t_vit_t_14_8xb64_in1k
model_dict = {
    'model_path':'D:/project/model_zoo/t2t_vit_t_14_8xb64_in1k/model.pt',
    'mode':'pt',
    'input_info':[
        {
            "input_name":"inputs",
            'bin_path':'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/test_model_input.bin',
            'input_shape':[1,3,224,224]
         },
    ]
}
target = "llvm"
save_model = True
save_dir = 'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/t2t_vit_t_14_8xb64_in1k/output'


# model_dict = {
#     'model_path':'D:/project/programs/my_project/tests/test_python/test_op/model_zoo3/index5/index5.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"v_0",
#             'bin_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/index/input0_4x2x3x4.bin',
#             'input_shape':[4,2,3,4]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/index/output'

# model_dict = {
#     'model_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/index/index.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/index/index_input.bin',
#             'input_shape':[1,3,7,56,228]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/index/output'

# model_dict = {
#     'model_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/permute/permute.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/permute/permute_input.bin',
#             'input_shape':[1,3,7,56,7,56]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/permute/output'


# model_dict = {
#     'model_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/reshape/reshape.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/reshape/reshape_input.bin',
#             'input_shape':[1,3,7,7,56,56]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/reshape/output'


# model_dict = {
#     'model_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/pad/pad.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/pad/pad_input.bin',
#             'input_shape':[1,3,224,224]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/pad/output'


#cat
# model_dict = {
#     # 'model_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/cat/cat.pt',
#     # 'mode':'pt',
#     # 'input_info':[
#     #     {
#     #         "input_name":"x",
#     #         'bin_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/cat/cat_input.bin',
#     #         'input_shape':[1,196,384]
#     #      },
#     # ]
    
#     'model_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/onnx/cat/cat_sim.onnx',
#     'mode':'onnx',
#     'input_info':[
#         {
#             "input_name":"input0",
#             'bin_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/onnx/cat/cat_input.bin',
#             'input_shape':[1,196,384]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/cat/output1'

# test model
# model_dict = {
#     'model_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/test_model/test_model.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/test_model/test_model_input.bin',
#             'input_shape':[1,3,224,224]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/tvm/run_model/mode_zoo/pt/test_model/output'

os.makedirs(save_dir, exist_ok= True)
load_model_flag = False

export_pt_tvm(model_dict, target, save_model, load_model_flag, save_dir)
# import torch
# import time
# import tvm
# from tvm import relay
# import numpy as np
# # input_array = np.fromfile(input_bin_path, dtype=np.float32).reshape(input_shape)
# input_array = torch.rand([1,3,224,224])
# img = input_array.numpy()

# target = "llvm"
# target_host = "llvm"
# dev = tvm.cpu(0)
# # with open(r'D:/project/programs/other_project/tvm_project/tvm_test_model/scnet50_torch21/model.json', "w") as fo:
# #     mod = tvm.ir.load_json(fo)
# mod = tvm.ir.load_json(r'D:/project/programs/other_project/tvm_project/tvm_test_model/scnet50_torch21/model.json')
# params =  tvm.runtime.load_param_dict_from_file(r'D:/project/programs/other_project/tvm_project/tvm_test_model/scnet50_torch21/params.params')
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, target_host=target_host, params=params)


# # ######################################################################
# # Execute the portable graph on TVM
# # ---------------------------------
# # Now we can try deploying the compiled model on target.
# from tvm.contrib import graph_executor

# m = graph_executor.GraphModule(lib["default"](dev))
# input_name = "input0"
# tvm_time_spent=[]
# torch_time_spent=[]
# n_warmup=5
# n_time=10
# # tvm_t0 = time.process_time()
# for i in range(n_warmup+n_time):
#     dtype = "float32"
#     # Set inputs
#     m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
#     tvm_t0 = time.time()
#     # Execute
#     m.run()
#     # Get outputs
#     tvm_output = m.get_output(0)
#     tvm_time_spent.append(time.time() - tvm_t0)
# # tvm_t1 = time.process_time()

# print(tvm_output)

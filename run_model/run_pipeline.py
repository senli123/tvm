
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





# axial50s failed
# model_dict = {
#     'model_path':'D:/project/model_zoo/axial50s/model.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/test_model_input.bin',
#             'input_shape':[1,3,224,224]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/axial50s/output'




# gfl_r50_fpn_1x_coco failed
# model_dict = {
#     'model_path':'D:/project/model_zoo/gfl_r50_fpn_1x_coco/model.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'D:/project/model_zoo/gfl_r50_fpn_1x_coco/input_1x3x1333x800.bin',
#             'input_shape':[1,3,1333,800]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/model_zoo/gfl_r50_fpn_1x_coco/output'

# GFNet onnx pass
# model_dict = {
#     'model_path':'D:/project/model_zoo/GFnet/GFNet_sim.onnx',
#     'mode':'onnx',
#     'input_info':[
#         {
#             "input_name":"input0",
#             'bin_path':'',
#             'input_shape':[1,1,181,217,181]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/onnx/GFNet/output'

# GFNet pt paild
# model_dict = {
#     'model_path':'D:/project/model_zoo/GFnet/GFNet.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'',
#             'input_shape':[1,1,181,217,181]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/GFNet/output'


#DDN pass
# model_dict = {
#     'model_path':'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/DDN/DDN.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"HTy",
#             'bin_path':'',
#             'input_shape':[1,1,320,320]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# save_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/DDN/output'


# mvitv2 failed
# model_dict = {
#     'model_path':'D:/project/model_zoo/mvitv2/model.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'',
#             'input_shape':[1,3,224, 224]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# model_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/mvitv2'

# PASSRnet failed
# model_dict = {
#     'model_path':'D:/project/model_zoo/PASSRnet/model.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x_left",
#             'bin_path':'',
#             'input_shape':[1,3,512, 512]
#          },
#          {
#             "input_name":"x_right",
#             'bin_path':'',
#             'input_shape':[1,3,512, 512]
#          }
#     ]
# }
# target = "llvm"
# save_model = True
# model_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/PASSRnet' #Y:/trans_onnx/cvpr2019/PASSRnet-master/PASSRnet-master 中间有代码是拿numpy去写的必须用torch.export图去解析

# postfilter
# model_dict = {
#     'model_path':'D:/project/model_zoo/postfilter/test_img/out111/model.pt',
#     'mode':'pt',
#     'input_data_dict_path': 'D:/project/programs/ncnn_project/nvppnnx/model_zoo/postfilter_3/input_tensor_container.pt',
#         'input_info':[
#          {
#             "input_name":"box",
#             'bin_path':'',
#             'input_shape':[1,4,1,60]
#          },
#           {
#             "input_name":"score",
#             'bin_path':'',
#             'input_shape':[1,1,1,60]
#          },
#            {
#             "input_name":"label",
#             'bin_path':'',
#             'input_shape':[1,1,1,60]
#          },
#         {
#             "input_name":"mask",
#             'bin_path':'',
#             'input_shape':[1,60,64,64]
#          }
         
#     ]
# }
# model_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/postfilter_3'


# model_dict = {
#     'model_path':'D:/project/model_zoo/rtm/RtmPreprocess.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'',
#             'input_shape':[1,3,1878,1764]
#          },
#     ]
# }
# target = "llvm"
# save_model = True
# model_dir = 'D:/project/model_zoo/rtm/output'

# hair failed
# model_dict = {
#     'model_path':'D:/project/model_zoo/HAIR-main/hair.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'',
#             'input_shape':[1,3,128,128]
#          },
#     ]
# }
# model_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/HAIR-main'

# swinir
# model_dict = {
#     'model_path':'D:/project/model_zoo/SwinIR/swinir.pt',
#     'mode':'pt',
#     'input_info':[
#         {
#             "input_name":"x",
#             'bin_path':'D:/project/model_zoo/SwinIR/1x3x264x264.bin',
#             'input_shape':[1,3,264,264]
#          },
#     ]
# }
# model_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/SwinIR'


# test model
model_dict = {
    'model_path':'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/test_model.pt',
    'mode':'pt',
    'input_info':[
        {
            "input_name":"x",
            'bin_path':'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/test_model_input.bin',
            'input_shape':[1,3,224,224]
         },
    ]
}
model_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/output3'

os.makedirs(model_dir, exist_ok= True)
save_dir = model_dir + '/output'
os.makedirs(save_dir, exist_ok= True)

target = "llvm"
save_model = True
load_model_flag = False
dump = True
export_pt_tvm(model_dict, target, save_model, load_model_flag, save_dir, dump)
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

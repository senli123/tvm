from pipeline import export_pt_tvm
import os
# scnet50_torch21
# input_pt_path= 'Y:/trans_onnx/pnnx_test/scnet50_torch21/scnet50_torch21.pt'
# input_bin_path= 'Y:/trans_onnx/pnnx_test/scnet50_torch21/data_1x224x224x3.bin'
# input_shape = [1,3, 224, 224]
# target = "llvm"
# save_model = True
# save_dir = r'D:\project\programs\other_project\tvm_project\tvm_test_model\scnet50_torch21_2'


#
model_dict = {
    'model_path':'/workspace/trans_onnx/project/tvm/run_model/model_zoo/resnet18/resnet18.pt',
    'mode':'pt',
    'input_info':[
        {
            'bin_path':'/workspace/trans_onnx/project/tvm/run_model/model_zoo/resnet18/resnet18_input.bin',
            'input_shape':[1,3,224,224]
         },
    ]
}
target = "llvm"
save_model = True
save_dir = '/workspace/trans_onnx/project/tvm/run_model/model_zoo/resnet18/output'

os.makedirs(save_dir, exist_ok= True)
load_model_flag = True
export_pt_tvm(model_dict, target, save_model, load_model_flag, save_dir)
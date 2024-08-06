## 测试记录

### onnx

#### resnet

```python
#resnet18 onnx 
model_dict = {
    'model_path':'D:/project/programs/other_project/tvm_project/tvm_test_model/onnx/resnet18/resnet18_sim.onnx',
    'mode':'onnx',
    'input_info':[
        {
            'bin_path':'D:/project/programs/other_project/tvm_project/tvm_test_model/onnx/resnet18/resnet18_input.bin',
            'input_shape':[1,3,224,224]
         },
    ]
}
target = "llvm"
save_model = False
save_dir = 'D:/project/programs/other_project/tvm_project/tvm_test_model/onnx/resnet18/output'

os.makedirs(save_dir, exist_ok= True)
load_model_flag = False
```


#### whisper

```python
#whisper 
model_dict = {
    'model_path':'D:/project/model_zoo/whisper_tiny/whisper-encoder_sim.onnx',
    'mode':'onnx',
    'input_info':[
        {
            "input_name":"x.1",
            'bin_path':'D:/project/programs/other_project/tvm_project/tvm_test_model/onnx/whisper-encoder_sim/input_1x80X3000.bin',
            'input_shape':[1,80,3000]
         },
    ]
}
target = "llvm"
save_model = True
save_dir = 'D:/project/programs/other_project/tvm_project/tvm_test_model/onnx/whisper-encoder_sim/output'

os.makedirs(save_dir, exist_ok= True)
load_model_flag = False
```


### pt

#### mmpose_rtmo

```python
# mmpose_rtmo
model_dict = {
    # 'model_path':'D:/project/model_zoo/mmpose_rtmo/model.pt',
    'model_path':'D:/project/model_zoo/rtmo-s/end2end_1.pt',
    'mode':'pt',
    'input_info':[
        {
            "input_name":"x",
            'bin_path':'D:/project/programs/other_project/tvm_project/tvm_test_model/pt/mmpose_rtmo/input_1x3x640x640.bin',
            'input_shape':[1,3,640,640]
         },
    ]
}
target = "llvm"
save_model = True
save_dir = 'D:/project/programs/other_project/tvm_project/tvm_test_model/pt/mmpose_rtmo/output'

os.makedirs(save_dir, exist_ok= True)
load_model_flag = False
```

#### vig_tiny_3rdparty_in1k
```python
# vig_tiny_3rdparty_in1k
model_dict = {
    'model_path':'D:/project/model_zoo/vig_tiny_3rdparty_in1k/model.pt',
    'mode':'pt',
    'input_info':[
        {
            "input_name":"inputs",
            'bin_path':'D:/project/programs/other_project/tvm_project/tvm_test_model/pt/vig_tiny_3rdparty_in1k/input_1x3x224x224.bin',
            'input_shape':[1,3,224,224]
         },
    ]
}
target = "llvm"
save_model = True
save_dir = 'D:/project/programs/other_project/tvm_project/tvm_test_model/pt/vig_tiny_3rdparty_in1k/output1'

os.makedirs(save_dir, exist_ok= True)
load_model_flag = False

```


#### t2t_vit_t_14_8xb64_in1k
```python
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
```
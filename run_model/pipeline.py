
import os
from typing import List
import time
import numpy as np
import torch 
import torchvision
import onnx
import tvm
import scipy.spatial.distance as dist
import platform 
from tvm import relay
import onnxruntime as rt
import sys
import pickle
def simlarity(torch_outs, tvm_outs):
    assert len(torch_outs) == len(tvm_outs), "torch_out must equal to tvm_out"
    for index, (torch_out, tvm_out) in enumerate(zip(torch_outs, tvm_outs)):
        print("===========output_index :{}=============".format(index))
        print("torch shape:{}, tvm shape:{}".format(torch_out.shape, tvm_out.shape))
        print("torch min:{}, tvm min:{}".format(np.min(torch_out), np.min(tvm_out)))
        print("torch max:{}, tvm max:{}".format(np.max(torch_out), np.max(tvm_out)))
        simlarity_value = 1 - dist.cosine(torch_out.ravel().astype(np.float32), tvm_out.ravel().astype(np.float32))
        print("simlarity: {}".format(simlarity_value))
            

def save_llvm_ir(mod, target, mod_name):
    """保存LLVM IR的回调函数"""
    llvm_ir = mod.get_text(show_meta_data=False)
    with open(f"{mod_name}_llvm_ir_{target}.ll", "w") as f:
        f.write(llvm_ir)

def save_model(mod, params, lib, extension, save_dir):
    """_summary_

    Args:
        mod (_type_): _description_
        params (_type_): _description_
        lib (_type_): _description_
        save_dir (_type_): _description_

    other type
        temp = util.tempdir()
        path_lib = temp.relpath("deploy_lib.tar")
        lib.export_library(path_lib)
        with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph)
        with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(params))
        print(temp.listdir())
    """
    save_model_path = os.path.join(save_dir, 'model.json')
    save_param_path = os.path.join(save_dir, 'params.params')
    if platform.system() == "Windows":  
        save_lib_path = os.path.join(save_dir, 'lib.dll')
    elif platform.system() == "Linux": 
        save_lib_path = os.path.join(save_dir, 'lib.so')
    with open(save_model_path, "w") as fo:
        fo.write(tvm.ir.save_json(mod))
        

    with open(save_param_path, "wb") as fo:
        fo.write(tvm.runtime.save_param_dict(params))

    lib.export_library(save_lib_path) 

    # 使用 get_source 方法获取源代码
    source_code = lib.get_lib().get_source()

    # 定义要保存的文件路径
    file_path = os.path.join(save_dir, "compiled_code." + extension)  # 对于 LLVM，通常使用 .ll 扩展名

    # 将源代码保存到文件中
    with open(file_path, "w") as file:
        file.write(source_code)

    print(f"源代码已保存到 {file_path}")

    
def load_model(save_dir):
    """_summary_

    Args:
        save_model_path (_type_): _description_
        save_param_path (_type_): _description_
        save_lib_path (_type_): _description_

    Returns:
        _type_: _description_

    other type
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_lib = tvm.module.load(path_lib)
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))
    """
    save_model_path = os.path.join(save_dir, 'model.json')
    save_param_path = os.path.join(save_dir, 'params.params')
    if platform.system() == "Windows":  
        save_lib_path = os.path.join(save_dir, 'lib.dll')
    elif platform.system() == "Linux": 
        save_lib_path = os.path.join(save_dir, 'lib.so')

    with open(save_model_path, 'r') as fi:  
        json_str = fi.read()  
        mod = tvm.ir.load_json(json_str)
    # with np.load(save_param_path, allow_pickle=True) as data:  
    #     loaded_params = {key: data[key] for key in data}
    with open(save_param_path, "rb") as fo:
        param_str = fo.read()
        loaded_params = tvm.runtime.load_param_dict(param_str)

    loaded_lib = tvm.runtime.load_module(save_lib_path)

    return mod, loaded_params, loaded_lib


def dump_pt_data(model, ptpath, input_data, pt_dump_root):

    def createTuple(graph2: torch.Graph, values:List[torch.Value]) -> torch.Node:
        """_summary_
            refer to
            torch\csrc\jit\ir\ir.cpp  Node* Graph::createTuple(at::ArrayRef<Value*> values, TupleTypePtr tuple_type)
        Args:
            values (List[torch.Value]): _description_

        Returns:
            torch.Node: _description_
        """
        types = [v.type() for v in values]
        tuple_type = torch.TupleType(types)
        n : torch.Node = graph2.create('prim::TupleConstruct', values, 1) 
        n.output().setType(tuple_type)
        return n
    
    dump_pt_data = {}
    
    graph1 :torch._C.Graph = model.graph.copy()
    names1 = []
    for cur_node1 in graph1.nodes():
        for cur_output1 in cur_node1.outputs():
            tensor_type = cur_output1.type()
            if(not tensor_type.isSubtypeOf(torch.TensorType.get())):
                continue
            names1.append(cur_output1.debugName())
    
    mod2:torch.jit.ScriptModule = torch.load(ptpath)
    # mod2 = model.copy()
    mod2.eval()

    graph2 :torch._C.Graph = mod2.graph
   
    values2: List[torch.Value] = []
    names = []
    for cur_node2 in graph2.nodes():
        for cur_output2 in cur_node2.outputs():
            tensor_type = cur_output2.type()
            if(not tensor_type.isSubtypeOf(torch.TensorType.get())):
                continue
            values2.append(cur_output2)
            names.append(cur_output2.debugName())
    
     # set new graph output
    new_return_node = createTuple(graph2, values2)

    graph2.appendNode(new_return_node)

    graph2.eraseOutput(0)
    graph2.registerOutput(list(new_return_node.outputs())[0])
    
    outputs = mod2.forward(*input_data)
    for out, v, scr_name, dst_name in zip(outputs, values2, names, names1):
        t2 = out.detach().numpy()
        value_name = v.debugName()
        assert value_name == scr_name
        dump_pt_data[dst_name] = t2
    
    pickle_file_path  = os.path.join(pt_dump_root, 'data.pkl')
    with open(pickle_file_path, 'wb') as f:  
        pickle.dump(dump_pt_data, f)  
    
    return list(dump_pt_data.keys())
    
    
def export_pt_tvm(model_dict, target, save_model_flag = False, load_model_flag = False, save_dir = '', dump = False):
    # parse model_dict
    model_path = model_dict['model_path']
    mode = model_dict['mode']
    input_info = model_dict['input_info']
    if mode == 'pt':
        model = torch.load(model_path)
        model.eval()
    elif mode == 'onnx':
        model = onnx.load(model_path)
    else:
        assert False, 'only support pt and onnx'
    shape_list = []
    image_list = []
    if mode == 'pt' and 'input_data_dict_path' in model_dict:
        input_data_dict_path = model_dict['input_data_dict_path']
        tensors = torch.jit.load(input_data_dict_path)
        if hasattr(tensors,'input_tensors'):
            input_tensors = tensors.input_tensors
            if not isinstance(input_tensors, list):
               assert False, "please check input data" 
        else:
            assert False, "please check input data"
    else:
        input_data_dict_path = None
    for index, input_info_dict in enumerate(input_info):
        input_name = input_info_dict['input_name']
        input_bin_path = input_info_dict['bin_path']
        input_shape = input_info_dict['input_shape']
        # prepare input data
        if input_data_dict_path != None:
             input_array = input_tensors[index].numpy()
        elif not os.path.exists(input_bin_path):
            input_array = np.random.uniform(size = input_shape).astype(np.float32)
        else:
            input_array = np.fromfile(input_bin_path, dtype=np.float32).reshape(input_shape)
        # if mode == 'pt':
        #     input_name = 'input' + str(index)
        # elif mode == 'onnx':
        #     input_name = 'input.' + str(index+1)
        # input_name = 'input' + str(index)
        shape_list.append((input_name,input_array.shape))
        image_list.append(input_array)

    if target == "llvm":
        target = "llvm"
        target_host = "llvm"
        extension = "ll"
        dev = tvm.cpu(0)
    elif target == "c":
        target = "c"
        target_host = "c"
        dev = tvm.cpu(0)
        extension = 'c'
    elif target == "cuda":
        target = tvm.target.cuda()
        target_host = "llvm"
        dev = tvm.device(str(target), 0)
        extension = 'cu'
    if load_model_flag:   
        mod, params, lib = load_model(save_dir)

    else:
        ######################################################################
        # Import the graph to Relay
        # -------------------------
        # Convert PyTorch graph to Relay graph. The input name can be arbitrary.
        if mode == 'pt':
            mod, params, dump_tensor_names, dump_node_name_dict = relay.frontend.from_pytorch(model, shape_list, dump = dump)
        elif mode == 'onnx':
            shape_dict = {}
            for input_name, input_shape in shape_list:
                shape_dict[input_name] = input_shape
            mod, params = relay.frontend.from_onnx(model, shape_dict)
        ######################################################################
        # Relay Build
        # -----------
        # Compile the graph to llvm target with given input specification.
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)

    # ######################################################################
    # Execute the portable graph on TVM
    # ---------------------------------
    # Now we can try deploying the compiled model on target.
    if dump  and load_model_flag: 
        from tvm.contrib import graph_executor
        dump_root = os.path.join(save_dir, 'tvmdbg')
        os.makedirs(dump_root, exist_ok= True)
        m = graph_executor.create(lib["get_graph_json"](), lib, dev, dump_root=dump_root)
        
    elif dump and not load_model_flag:
        
        from tvm.contrib.debugger.debug_executor import GraphModuleDebug
        dump_root = os.path.join(save_dir, 'tvmdbg')
        os.makedirs(dump_root, exist_ok= True)
        m = GraphModuleDebug(lib["debug_create"]("default", dev), [dev], lib.graph_json, dump_root=dump_root)
        
    else:
        
        
        from tvm.contrib import graph_executor
        m = graph_executor.GraphModule(lib["default"](dev))

    # cal spend time 
    # tvm_time_spent=[]
    # torch_time_spent=[]
    # n_warmup=5
    # n_time=10
    # # tvm_t0 = time.process_time()
    # for i in range(n_warmup+n_time):
    #     dtype = "float32"
    #     # Set inputs
    #     for shape_info, img in zip(shape_list, image_list):
    #         m.set_input(shape_info[0], tvm.nd.array(img.astype(dtype)))
    #     tvm_t0 = time.time()
    #     # Execute
    #     m.run()
    #     # Get outputs
    #     tvm_output = m.get_output(0)
    #     tvm_time_spent.append(time.time() - tvm_t0)
    # # tvm_t1 = time.process_time()
    
    # tvm infer and get output
    dtype = "float32"
    # Set inputs
    for shape_info, img in zip(shape_list, image_list):
        m.set_input(shape_info[0], tvm.nd.array(img.astype(dtype)))
    # Execute
    m.run()
    # Get outputs
    tvm_out = []
    output_num = m.get_num_outputs()
    for i in range(output_num):
        tvm_output = m.get_output(i)
        tvm_out.append(tvm_output.asnumpy())

    # ########################################################################
    if not dump:
        # compare
        if mode == 'pt':
            image_list = [torch.from_numpy(image) for image in  image_list]
            torch_out = model(*image_list)
            if isinstance(torch_out, list) or isinstance(torch_out, tuple):
                numpy_out = [single_out.detach().numpy() for single_out in torch_out]
            else:
                numpy_out = [torch_out.detach().numpy()]
        
        elif mode == 'onnx':
            sess = rt.InferenceSession(model_path) 
            input_dict = {}
            for index, image_array in enumerate(image_list):
        
                input_name = sess.get_inputs()[index].name 
                input_dict[input_name] = image_array
            output_name = sess.get_outputs()[0].name  

            # 准备输入数据  
            numpy_out = sess.run(None, input_dict)[0]
        simlarity(numpy_out, tvm_out)
    else:
        assert mode == 'pt', 'only support torchscript dump data'
        pt_dump_root = os.path.join(save_dir, 'ptdbg')
        os.makedirs(pt_dump_root, exist_ok= True)
        pickle_file_path  = os.path.join(pt_dump_root, 'data.pkl')
        json_file_path  = os.path.join(pt_dump_root, 'tensor_name.json')
        image_list = [torch.from_numpy(image) for image in  image_list]
        torch_out = model(*image_list)
        
        dump_pt_data = {}
        for tensor_name, out in zip(dump_tensor_names, torch_out):
            dump_pt_data[tensor_name] = out
        
        with open(pickle_file_path, 'wb') as f:  
            pickle.dump(dump_pt_data, f)  

        import json  

        dump_info = {  
            "dump_tensor_names": dump_tensor_names,  
            "dump_node_name_dict":dump_node_name_dict
        }  
        
        with open(json_file_path, 'w', encoding='utf-8') as f:  
          
            json.dump(dump_info, f, indent=4) 
        
   
    
    ######################################################################
    if( (not load_model_flag) and save_model_flag):
        save_model(mod, params, lib, extension, save_dir)



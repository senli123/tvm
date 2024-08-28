
import os
import torch
import tvm
from tvm import relay
import pickle 
import json
import numpy as np
import ast 
from typing import List, Dict, Optional,Union

OperandTypeDict = {
    'float32': np.float32,
    torch.float32: np.float32
}
class relayOp:
    def __init__(self, node_name: str, node_type: str, node_attrs: Dict, node_params: Dict = {}, input_nodes: List[str] = [], \
        output_nodes: List[str] = [], input_operands: List[str] = [], output_operands: List[str] = []):
        self.name = node_name
        self.type = node_type
        self.attrs = node_attrs
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.input_operands = input_operands
        self.output_operands = output_operands
        self.params = node_params
    

    
class relayOperand:
    def __init__(self, operand_name: str, operand_type: Union[str, torch.dtype, np.dtype], operand_shape: Union[torch.Size, List[int]], input_nodes: list[str] = [], output_nodes: list[str] = []):
        self.name = operand_name
        if isinstance(operand_type, str) or isinstance(operand_type, torch.dtype):
            self.type = OperandTypeDict[operand_type]
        else:
            self.type = operand_type
        if isinstance(operand_shape, torch.Size):
            self.shape = list(operand_shape)
        else:
            self.shape = operand_shape
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    
def parse_op_attr(other_info: str) -> Dict:   
    """
    sample: input:'name_hint: x, shape: (1, 3, 224, 224), dtype: float32'
            output = {'name_hint': 'x', 'shape': [1,3,224,224], 'dtype': float32}
    """ 
    
    def split_by_last_comma(s):  
        last_comma_index = s.rfind(',')  
        if last_comma_index != -1:  
            return s[:last_comma_index], s[last_comma_index+1:]  
        else:  
            return s, ""
    attr_dict = {} 
    info_list = other_info.split(': ')
    fina_info_list = []
    for info in info_list:
        s1, s2 = split_by_last_comma(info)
        if s1 != '':
            fina_info_list.append(s1.strip())
        if s2 != '':
            fina_info_list.append(s2.strip())

    assert len(fina_info_list) %2 == 0
    key_num = len(fina_info_list) // 2 
    for i in range(key_num):
        key_index = i * 2
        value_index = i * 2 + 1
        key_name = fina_info_list[key_index]
        value_str = fina_info_list[value_index]
        if (value_str.startswith('(') and value_str.endswith(')')) or  (value_str.startswith('[') and value_str.endswith(']')):
            value = ast.literal_eval(value_str)
            attr_dict[key_name] = value
        else:
            if key_name == 'dtype':
                attr_dict[key_name] = OperandTypeDict[value_str]
            else:
                attr_dict[key_name] = value_str
            
       
    
    return attr_dict
            

def UnionOp(name2OperatorDict, name2OperandDict, oname2operatorsDict):
    def UnionTwoOp(name2OperatorDict, name2OperandDict, oname2operatorsDict, op_name1, op_name2, operand_name):
        # union op
        #op1_attr to op2_attr
        name2OperatorDict[op_name2].attrs.update(name2OperatorDict[op_name1].attrs)
        #op1_params to op2_params
        name2OperatorDict[op_name2].params.update(name2OperatorDict[op_name1].params)
        # update op2 output nodes and output operands
        name2OperatorDict[op_name2].output_nodes = name2OperatorDict[op_name1].output_nodes
        name2OperatorDict[op_name2].output_operands = name2OperatorDict[op_name1].output_operands
        #sink nodes
        sink_nodes = name2OperatorDict[op_name1].output_nodes
        for sink_node in sink_nodes:
            name2OperatorDict[sink_node].input_nodes = name2OperatorDict[op_name1].input_nodes
        
        # union operand
        name2OperandDict[operand_name].output_nodes.remove(op_name1)
        # update oname2operatorsDict
        oname2operatorsDict[operand_name] = name2OperatorDict[op_name2]
        
        # delete op_name1
        del name2OperatorDict[op_name1]

    
    UnionType = [['nn.conv2d', 'nn.bias_add']]
    
    while True:
        find_union = False
        for op_name, op in name2OperatorDict.items():
            input_operands = op.input_operands
            output_operands  = op.output_operands
            if input_operands == output_operands:
                cur_node_type = op.type
                pre_node = op.input_nodes[0]
                pre_node_type = name2OperatorDict[pre_node].type
                for union_pre_node_type, union_cur_node_type in UnionType:
                    if cur_node_type == union_cur_node_type and pre_node_type == union_pre_node_type:
                        find_union = True
                        # to union op
                        UnionTwoOp(name2OperatorDict, name2OperandDict, oname2operatorsDict, op_name,pre_node, input_operands[0])
                        break
                if find_union:
                    break
        if not find_union:
            break                
            

def get_node_attr(node):
    # 将dense节点填充绿色矩形
    if "nn.dense" in node.type_name:
        return {"fillcolor": "green",
                "style": "filled"}
    # 将axis=-1的softmax节点填充橙色
    if "nn.softmax" in node.type_name and "axis: -1" in node.detail: 
        return {"fillcolor": "orange", 
                "style": "filled"}
    # 设置Var节点为椭圆形
    if "Var" in node.type_name:
        return {"shape": "ellipse"}
    return {"shape": "box"}



def parse_Relay2Dict(src2dst, dst2src, node_dict, relay_param, dump_tensor_names, dump_node_name_dict, dump_pt_data): 
    """_summary_

    Args:
        src2dst (_type_): _description_
        dst2src (_type_): _description_
        node_dict (_type_): _description_
        
    return:
        name2OperatorDict = dict()     #dict{name(str):operator(Operator)}
        name2OperandDict = dict()      #dict{name(str):operand(Operand)}
        name2InputOpDict = dict()      #dict{name(str):operator(Operator)}
        name2OutputOpDict = dict()     #dict{name(str):operator(Operator)}
        oname2operatorsDict = dict()   #dict{name(str):operator(Operator)}
        outTnsrNameLst = []
        inputTnsrNameLst = []
    """
    
    name2OperatorDict = dict() 
    name2OperandDict = dict()
    name2InputOpDict = dict()
    name2OutputOpDict = dict()
    oname2operatorsDict = dict()
    outTnsrNameLst = []
    inputTnsrNameLst = []
    
    for node_name, node in node_dict.items():
        node_type = node.type
        if node_type == 'Var(Input)':
            #get output nodes
            output_nodes = src2dst[node_name]
            #create output operands
            input_operand_attrs = parse_op_attr(node.other_info)
            input_operand = relayOperand(input_operand_attrs['name_hint'],input_operand_attrs['dtype'], input_operand_attrs['shape'], [node_name])
            op = relayOp(node_name, node_type, input_operand_attrs, output_nodes = output_nodes, output_operands = [input_operand_attrs['name_hint']])
            name2OperandDict[input_operand_attrs['name_hint']] = input_operand
            name2OperatorDict[node_name] = op
            name2InputOpDict[node_name] = op
            inputTnsrNameLst.append(input_operand_attrs['name_hint'])
        elif node_type == 'Func':
            cur_node_attrs = parse_op_attr(node.other_info)
            # -----------create Operand ---------------
            # get input node list
            input_nodes = dst2src[node_name]
            input_operands = []
            for input_node_name in input_nodes:
                #get input node
                input_node = node_dict[input_node_name]
                assert input_node.type.startswith('Call'), "The type of pre node of output must be Call node"
                if input_node_name in name2OperatorDict:
                    #pre_node add new output node
                    name2OperatorDict[input_node_name].output_nodes.append(node_name)
                    #get span info
                    span_info = input_node.span_info
                    #get cur operand name
                    cur_operand_names = dump_node_name_dict[span_info]
                    assert len(cur_operand_names) == 1, 'Now only support one output'
                    cur_operand_name = cur_operand_names[0]
                    if cur_operand_name in name2OperandDict:
                        name2OperandDict[cur_operand_name].output_nodes.append(node_name)
                    else:
                        # create new operand
                        new_input_operand_dtype = dump_pt_data[cur_operand_name].dtype
                        new_input_operand_shape = dump_pt_data[cur_operand_name].shape
                        new_input_operand = relayOperand(cur_operand_name, new_input_operand_dtype, new_input_operand_shape, [input_node_name], [node_name])
                        name2OperandDict[cur_operand_name] = new_input_operand
                        oname2operatorsDict[cur_operand_name] = name2OperatorDict[input_node_name]
                    name2OperatorDict[input_node_name].output_operands.append(cur_operand_name)    
                    input_operands.append(cur_operand_name)
                    outTnsrNameLst.append(cur_operand_name)
                else:
                    assert False, "Pre node must be registered"
                    
            new_op = relayOp(node_name, node_type, cur_node_attrs, {}, \
                input_nodes = input_nodes, output_nodes = [],\
                input_operands = input_operands,
                output_operands = [])     
            name2OperatorDict[node_name] = new_op
            name2OutputOpDict[node_name] = new_op
            
        elif node_type.startswith('Call'):
            params = {}
            cur_node_attrs = parse_op_attr(node.other_info)
            call_node_type = node_type.split(' ')[1]
            input_nodes = dst2src[node_name]
            rel_input_nodes = []
            output_nodes = src2dst[node_name]
            input_operands = []
            for input_node_name in input_nodes:
                #get input node
                input_node = node_dict[input_node_name]
                if input_node.type == call_node_type:
                    call_node_type = input_node.type
                elif input_node.type == 'Var(Param)':
                    input_params_attrs = parse_op_attr(input_node.other_info)
                    input_params_name = input_params_attrs['name_hint']
                    params_data = relay_param[input_params_name].asnumpy()
                    params_info = {'data': params_data, 'shape': input_params_attrs['shape'], 'dtype':input_params_attrs['dtype']}
                    params[input_params_name] = params_info
                elif input_node.type == 'Var(Input)':
                    rel_input_nodes.append(input_node_name)
                    pre_node_output_operands = name2InputOpDict[input_node_name].output_operands
                    input_operands.extend(pre_node_output_operands)
                    for pre_node_output_operand in pre_node_output_operands:
                        name2OperandDict[pre_node_output_operand].output_nodes.append(node_name)
                else:
                    rel_input_nodes.append(input_node_name)
                    if input_node_name in name2OperatorDict:
                        #pre_node add new output node
                        name2OperatorDict[input_node_name].output_nodes.append(node_name)
                        #get cur operand name
                        cur_operand_name = dump_node_name_dict[input_node.span_info][0]
                        if cur_operand_name in name2OperandDict:
                            name2OperandDict[cur_operand_name].output_nodes.append(node_name)
                        else:
                            # create new operand
                            new_input_operand_dtype = dump_pt_data[cur_operand_name].dtype
                            new_input_operand_shape = dump_pt_data[cur_operand_name].shape
                            new_input_operand = relayOperand(cur_operand_name, new_input_operand_dtype, new_input_operand_shape, [input_node_name], [node_name])
                            name2OperandDict[cur_operand_name] = new_input_operand
                            oname2operatorsDict[cur_operand_name] = name2OperatorDict[input_node_name]
                        name2OperatorDict[input_node_name].output_operands.append(cur_operand_name)    
                        input_operands.append(cur_operand_name)
                    else:
                        assert False, "Pre node must be registered"
                    
            new_op = relayOp(node_name, call_node_type, cur_node_attrs, params, \
                input_nodes = rel_input_nodes, output_nodes = [],\
                input_operands = input_operands,
                output_operands = []) 

            name2OperatorDict[node_name] = new_op
        else:
            print('find a op, type: {}, name: {}'.format(node_type, node_name))

    UnionOp(name2OperatorDict, name2OperandDict, oname2operatorsDict)
    
    return  name2OperatorDict, name2OperandDict, name2InputOpDict, name2OutputOpDict, oname2operatorsDict,\
        outTnsrNameLst, inputTnsrNameLst 
        
    
    

def get_relay(model_dict, save_dir, dump, load_model, visual):
    # parse model_dict
    model_path = model_dict['model_path']
    mode = model_dict['mode']
    input_info = model_dict['input_info']
    
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


    if load_model:
        save_model_path = os.path.join(save_dir, 'model.json')
        save_param_path = os.path.join(save_dir, 'params.params')
        save_tensor_name_path = os.path.join(save_dir, 'ptdbg', 'tensor_name.json')
        save_data_path = os.path.join(save_dir,'ptdbg', 'data.pkl')
        # load mod
        with open(save_model_path, 'r') as fi:  
            json_str = fi.read()  
            mod = tvm.ir.load_json(json_str)

        # load params
        with open(save_param_path, "rb") as fo:
            param_str = fo.read()
            params = tvm.runtime.load_param_dict(param_str)
            
        # load dump_tensor_names, dump_node_name_dict
        with open(save_tensor_name_path, 'r') as f:  
            # 使用json.load()读取文件内容并反序列化为Python对象  
            data = json.load(f)  

        dump_tensor_names = data['dump_tensor_names']
        dump_node_name_dict = data['dump_node_name_dict']

        # load tensors 
        with open(save_data_path, 'rb') as f:  
            dump_pt_data = pickle.load(f)  
       
    else:
        
        model = torch.load(model_path)
        mod, params, dump_tensor_names, dump_node_name_dict = relay.frontend.from_pytorch(model, shape_list, dump = dump)
        
        
        image_list = [torch.from_numpy(image) for image in  image_list]
        torch_out = model(*image_list)
        
        dump_pt_data = {}
        for tensor_name, out in zip(dump_tensor_names, torch_out):
            dump_pt_data[tensor_name] = out
            
    
    from tvm.contrib import relay_viz
    
    
    # 获取vgg网络的IRModule和param
    # mod, param = mlp.get_workload(batch_size=1, num_classes=10)

    # graphviz属性
    graph_attr = {"color": "red"}
    node_attr = {"color": "blue"}
    edge_attr = {"color": "black"}
    # 创建一下plotter
    dot_plotter = relay_viz.DotPlotter(
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
        get_node_attr=get_node_attr)

    viz = relay_viz.RelayVisualizer(
        mod,
        relay_param=params,
        plotter=dot_plotter,   # 传入定义的plotter
        parser=relay_viz.DotVizParser(),
        visual_mod = 1)
    
    graph, relay_param = viz.get_graph_info()
    src2dst = graph.src2dst
    dst2src = graph._graph
    node_dict = graph._id_to_term_node
            
    name2OperatorDict, name2OperandDict, name2InputOpDict, name2OutputOpDict, oname2operatorsDict,\
        outTnsrNameLst, inputTnsrNameLst  = parse_Relay2Dict(src2dst, dst2src, node_dict, relay_param, dump_tensor_names, dump_node_name_dict, dump_pt_data)

    if visual:
        from visual_test import visual_relay,visual_relay_to_ncnn
        # visual_relay(name2OperatorDict, name2OperandDict, save_dir)
        visual_relay_to_ncnn(name2OperatorDict, name2OperandDict, save_dir)

  
if __name__ == "__main__":
    # ----------------get relay-----------
    # sample 1
    model_dict = {
    'model_path':'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/test_model.pt',
    'mode':'pt',
    'input_info':[
        {
            "input_name":"x",
            'bin_path':'',
            'input_shape':[1,3,224,224]
         },
    ]
    }
    save_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/output3/output'
    dump = True
    load_model = True 
    visual = True
    # sample 2
    
    # model_path = '/workspace/my_tvm/model_zoo/pt/test_model/test_model.pt'
    # save_dir = '/workspace/my_tvm/model_zoo/pt/test_model'
    # shape_list = [('x', [1,3,224,224])]
    # dump = False
    # load_model = False 
    
    get_relay(model_dict, save_dir, dump, load_model, visual)
    
   
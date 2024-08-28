""" the sample of visual relay by graphviz
"""
import platform
import os
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

def visual_relay_by_graphviz(mod, param):
    
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
        relay_param=param,
        plotter=dot_plotter,   # 传入定义的plotter
        parser=relay_viz.DotVizParser(),
        visual_mod = 1)

    # 渲染生成 pdf
    viz.render("mlp")
    
""" the sample of visual relay by netron
"""
import json
import numpy as np
from typing import Dict
from tvm.contrib.relay_viz.interface import (
    DefaultVizParser, Plotter,
    VizEdge, VizGraph, VizNode,
)

try:
    import h5py
    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    # add "from None" to silence
    # "During handling of the above exception, another exception occurred"
    raise ImportError(
        "The h5py package is required. "
        "Please install it first. For example, pip3 install h5py"
    ) from None

Hdf5VizParser = DefaultVizParser

class Hdf5Node:
    def __init__(self, viz_node: VizNode):
        self.name = viz_node.type_name + '_' + viz_node.identity
        self.type = viz_node.type_name
        self.params = self._detail_to_params(viz_node.detail)

    def _detail_to_params(self, detail: str) -> Dict:
        if len(detail) == 0:
            return {}
        if platform.system() == "Windows":  
            ds = detail.split("\n")
        elif platform.system() == "Linux": 
            ds = detail.split("/n")
        params = {}
        for p in ds:
            k, v = p.split(": ")
            params[k] = v
        return params

class Hdf5Graph(VizGraph):
    """Hdf5 graph for relay IR.
    Parameters
    ----------
    name: str
        name of this graph.
    """

    def __init__(
        self,
        name: str
    ):
        self._name = name
        self._graph = {}
        self._id_to_hf_node = {}

    def node(self, viz_node: VizNode) -> None:
        """Add a node to the underlying graph."""

        if viz_node.identity not in self._graph:
            # Add the node into the graph.
            self._graph[viz_node.identity] = []

        node = Hdf5Node(viz_node)
        self._id_to_hf_node[viz_node.identity] = node

    def edge(self, viz_edge: VizEdge) -> None:
        """Add an edge to the underlying graph."""
        if viz_edge.end in self._graph:
            self._graph[viz_edge.end].append(viz_edge.start)
        else:
            self._graph[viz_edge.end] = [viz_edge.start]

    def get_layers(self):
        layers = []
        for id, in_ids in self._graph.items():
            layer = {}
            hf_node = self._id_to_hf_node[id]
            layer['name'] = hf_node.name
            layer['class_name'] = hf_node.type
            layer['inbound_nodes'] = []
            dtype_hint = ""
            for in_id in in_ids:
                in_hf_node = self._id_to_hf_node[in_id]
                item = [in_hf_node.name, 0, 0, {}]
                layer['inbound_nodes'].append(item)
                if 'dtype' in in_hf_node.params.keys():
                    dtype_hint = in_hf_node.params['dtype']

            layer['config'] = {'name': hf_node.params['name_hint'] if 'name_hint' in hf_node.params.keys() else hf_node.name}
            for k, v in hf_node.params.items():
                if 'out_dtype' in k and v and v == ' ':
                    layer['config']['dtype'] = dtype_hint
                    continue
                if 'name_hint' in k or not v or len(v) == 0:
                    continue
                layer['config'][k] = v

            layers.append(layer)
        return layers

class Hdf5Plotter(Plotter):
    """Hdf5 graph plotter"""

    def __init__(self):
        self._name_to_graph = {}

    def _save_attr_to_group(self, group, name, data):
        bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

        # Expecting this to never be true.
        if bad_attributes:
            raise RuntimeError('The following attributes cannot be saved to HDF5 '
                               'file because they are larger than %d bytes: %s' %
                               (HDF5_OBJECT_HEADER_LIMIT, ', '.join(bad_attributes)))

        data_npy = np.asarray(data)

        num_chunks = 1
        chunked_data = np.array_split(data_npy, num_chunks)

        # This will never loop forever thanks to the test above.
        while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
            num_chunks += 1
            chunked_data = np.array_split(data_npy, num_chunks)

        if num_chunks > 1:
            for chunk_id, chunk_data in enumerate(chunked_data):
                group.attrs['%s%d' % (name, chunk_id)] = chunk_data
        else:
            group.attrs[name] = data

    def create_graph(self, name):
        self._name_to_graph[name] = Hdf5Graph(name)
        return self._name_to_graph[name]

    def render(self, filename: str = None):
        for name in self._name_to_graph:
            if filename is None:
                filename = name
            f = h5py.File(filename + '.h5', mode='w')
            g = f.create_group('model_weights')
            g.attrs['backend'] = 'tvm.relay'.encode('utf8')
            g.attrs['tvm_version'] = "0.11".encode('utf8')

            layers = self._name_to_graph[name].get_layers()
            mod_cfg = {"class_name": "Model", "config": {"name": "model", 'layers': layers}}
            f.attrs['model_config'] = json.dumps(mod_cfg).encode('utf8')
            self._save_attr_to_group(g, 'layer_names', [layer['name'].encode('utf8') for layer in layers])
            f.close()
            
def visual_relay_by_netron(mod, param):
   
    from tvm.contrib import relay_viz

    # mod, param = mlp.get_workload(batch_size=1, num_classes=10)
    print("mod:{}/n/n".format(mod))

    viz = relay_viz.RelayVisualizer(
        mod,
        relay_param=param,
        plotter=Hdf5Plotter(),
        parser=Hdf5VizParser(),
        visual_mod = 0
        )
    viz.render('mlp')            


def visual_relay(name2OperatorDict, name2OperandDict, save_dir):
    
    # get 
    # 1. _graph = {} dict of dst_node_id: [src_node]
    # 2._id_to_hf_node = {} dict of node_id: node
    print(123)
    numpyTupe2Str = {np.float32: 'float32'}
    def get_layers(_graph, name2OperatorDict):
        layers = []
        for id, in_ids in _graph.items():
            layer = {}
            hf_node = name2OperatorDict[id]
            layer['name'] = hf_node.name
            layer['class_name'] = hf_node.type
            layer['inbound_nodes'] = []
            for in_id in in_ids:
                in_hf_node = name2OperatorDict[in_id]
                item = [in_hf_node.name, 0, 0, '123']
                layer['inbound_nodes'].append(item)
            layer['config'] = {}
            layer['config']['name'] = hf_node.name
            for k, v in hf_node.attrs.items():
                if k == 'dtype':
                    layer['config'][k] = numpyTupe2Str[v]
                    continue
                layer['config'][k] = v

            layers.append(layer)
        return layers
    
    def _save_attr_to_group(group, name, data):
        bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

        # Expecting this to never be true.
        if bad_attributes:
            raise RuntimeError('The following attributes cannot be saved to HDF5 '
                               'file because they are larger than %d bytes: %s' %
                               (HDF5_OBJECT_HEADER_LIMIT, ', '.join(bad_attributes)))

        data_npy = np.asarray(data)

        num_chunks = 1
        chunked_data = np.array_split(data_npy, num_chunks)

        # This will never loop forever thanks to the test above.
        while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
            num_chunks += 1
            chunked_data = np.array_split(data_npy, num_chunks)

        if num_chunks > 1:
            for chunk_id, chunk_data in enumerate(chunked_data):
                group.attrs['%s%d' % (name, chunk_id)] = chunk_data
        else:
            group.attrs[name] = data
            
    def render(save_dir, layers):
        file_name = os.path.join(save_dir, 'model.h5')
        f = h5py.File(file_name, mode='w')
        g = f.create_group('model_weights')
        g.attrs['backend'] = 'tvm.relay'.encode('utf8')
        g.attrs['tvm_version'] = "0.11".encode('utf8')
        mod_cfg = {"class_name": "Model", "config": {"name": "model", 'layers': layers}}
        f.attrs['model_config'] = json.dumps(mod_cfg).encode('utf8')
        _save_attr_to_group(g, 'layer_names', [layer['name'].encode('utf8') for layer in layers])
        f.close()
    
    _graph = {}
    for node_name, node in name2OperatorDict.items():
        _graph[node_name] = node.input_nodes
        
    layers = get_layers(_graph, name2OperatorDict)
    render(save_dir, layers)

def visual_relay_to_ncnn(name2OperatorDict, name2OperandDict, save_dir):
    numpyType2Str = {np.float32: 'float32'}
    file_name = os.path.join(save_dir, 'model.pnnx.param')
    with open(file_name, "w", encoding="utf-8") as f:
        f.write("7767517\n")
        f.write("{} {}\n".format(len(name2OperatorDict), len(name2OperandDict)))
        for node_name, node in name2OperatorDict.items():
            f.write("{:<24} {:<24} {} {}".format(node.type, node.name, len(node.input_operands), len(node.output_operands)))
            
            for input_operand in node.input_operands:
                f.write(" {}".format(name2OperandDict[input_operand].name))
                
            for output_operand in node.output_operands:
                f.write(" {}".format(name2OperandDict[output_operand].name))
            
            if node.type != 'Var(Input)' and node.type != 'Func': 
                for attr_name, attr_value in node.attrs.items():
                    f.write(" {}={}".format(attr_name, str(attr_value).replace(' ', '')))
                    
            for param_name, param_value in node.params.items():
                # get shape type
                shape_str = str(param_value['shape']).replace(' ', '')
                dtype_str = numpyType2Str[param_value['dtype']]
                f.write(" @{}={}".format(param_name, shape_str + dtype_str))
                
            for input_operand in node.input_operands:
                operand_name = name2OperandDict[input_operand].name
                operand_shape_str = str(name2OperandDict[input_operand].shape).replace(' ', '')
                operand_type_str = numpyType2Str[name2OperandDict[input_operand].type]
                f.write(" #{}={}".format(operand_name, operand_shape_str + operand_type_str))
                
            for output_operand in node.output_operands:
                operand_name = name2OperandDict[output_operand].name
                operand_shape_str = str(name2OperandDict[output_operand].shape).replace(' ', '')
                operand_type_str = numpyType2Str[name2OperandDict[output_operand].type]
                f.write(" #{}={}".format(operand_name, operand_shape_str + operand_type_str))
            f.write("\n")
        f.close()
        
if __name__ == "__main__":
    
    # from tvm.relay.testing import mlp
    # mod, param = mlp.get_workload(batch_size=1, num_classes=10)

    from parse_relay_ir import get_relay
    # model_path = '/workspace/my_tvm/model_zoo/pt/test_model/test_model.pt'
    # save_dir = '/workspace/my_tvm/model_zoo/pt/test_model'
    
    model_path = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/test_model.pt'
    save_dir = 'D:/project/programs/other_project/tvm_project/new_tvm/mode_zoo/pt/test_model/output'

    shape_list = [('x', [1,3,224,224])]
    dump = False
    load_model = False 
    
    # mod, param = get_relay(model_path, save_dir, shape_list, dump,  load_model)
    import torch
    from tvm import relay
    model = torch.load(model_path)
    mod, params, dump_tensor_names, dump_node_name_dict = relay.frontend.from_pytorch(model, shape_list, dump = dump)
        
    
    # visual relay by graphviz
    # visual_relay_by_graphviz(mod, param)
    #
    visual_relay_by_netron(mod, params)

import torch
import torch.nn as nn
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import os
from tvm.contrib import utils
def update_lib(lib):
    test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(test_dir, "..", "..", "..", "..")
    contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++17", "-I" + contrib_path]
    tmp_path = utils.tempdir()
    lib_name = "lib.dll"
    lib_path = tmp_path.relpath(lib_name)
    lib.export_library(lib_path, fcompile=False, **kwargs)
    lib = tvm.runtime.load_module(lib_path)

    return lib


class AddModel(nn.Module):
    def __init__(self,):
        super(AddModel, self).__init__()
        self.a = torch.ones([1,2])
    def forward(self, v_0):
        one_hot = v_0 + self.a
        return one_hot

model = AddModel()
model.eval()
input = torch.ones([1,2])
mod = torch.jit.trace(model, input)
shape_list = [('x', (1, 2))]
mod, params = relay.frontend.from_pytorch(mod, shape_list)
# print(mod)
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, target_host=target_host, params=params)

# mod = get_demo_mod()
mod = relay.transform.AnnotateTarget("ccompiler")(mod)
mod = relay.transform.PartitionGraph()(mod)
# print(mod)
with tvm.transform.PassContext(opt_level=2):
    graph, lib, params = relay.build(mod, target="llvm", params=None)
    # lib = relay.build(mod, target="c", target_host="llvm", params=params)
    # executor_factory = relay.build(mod, target="c", params=params)
    # graph, lib, params = relay.build(mod, "llvm", params=params)
# lib_kwargs = {}    
# lib.export_library("tvm_dpu_cpu.so", **lib_kwargs)
# lib.export_library("liba.so")
# dev = tvm.cpu(0)
# lib = update_lib(executor_factory.lib)
# rt_mod = graph_executor.create(executor_factory.graph_json, lib, dev)
for i in range(len(lib.imported_modules)):
    print("imported_modules id :{}".format(i))
    source_code = lib.imported_modules[i].get_source()
    file_path = os.path.join("compiled_code{}.cpp".format(i))  # 对于 LLVM，通常使用 .ll 扩展名
    # 将源代码保存到文件中
    with open(file_path, "w") as file:
        file.write(source_code)

    print(f"源代码已保存到 {file_path}")
# update_lib(lib)
lib.export_library("compiled_lib.dll")
# # load it back as a runtime
# lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_lib.dll")   
from tvm.contrib import graph_executor

m = graph_executor.GraphModule(lib["default"](dev))  

dtype = "float32"
# Set inputs
image_list = [input]
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
print(tvm_out[0])  
# from tvm import relay
# from tvm.contrib import graph_executor

# mod = create_relay_mod()
# from tvm.relay.op.contrib.codegen_c import partition_for_ccompiler
# pmod = partition_for_ccompiler(mod)  # do "ccompiler" annotation, also graph partitionning
# lib = relay.build(pmod, target)

# # Generate graph executor
# dev = tvm.device(target, 0)
# m = graph_executor.GraphModule(lib["default"](dev))

# dtype = 'float32'
# set_module_inputs(m)
# m.run()
# output = m.get_output(0)

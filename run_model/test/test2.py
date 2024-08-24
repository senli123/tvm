
path_params = r'D:\project\programs\other_project\tvm_project\new_tvm\mode_zoo\pt\test_model\output3\output\tvmdbg\_tvmdbg_device_CPU_0\output_tensors.params'
graph_dump_path  = r'D:\project\programs\other_project\tvm_project\new_tvm\mode_zoo\pt\test_model\output3\output\tvmdbg\_tvmdbg_device_CPU_0\_tvmdbg_graph_dump.json'
# with open(path_params, 'rb') as fi:
#     loaded_params = bytearray(fi.read())
# print(loaded_params)
import tvm
import json
with open(path_params, "rb") as fo:
    param_str = fo.read()
    loaded_params = tvm.runtime.load_param_dict(param_str)
    
print(loaded_params)

# Load graph trace + intermediate tensors
with open(path_params, "rb") as f:
    intermediate_tensors = dict(tvm.relay.load_param_dict(f.read()))

with open(graph_dump_path, "r") as f:
    graph = json.load(f)
    
print(graph)



import torch  
import torch.nn as nn
def pixel_unshuffle_custom(input, downscale_factor):  
    # 确保输入Tensor是四维的  
    batch_size, channels, height, width = input.size()  
      
    # 计算目标形状  
    new_channels = channels * downscale_factor ** 2  
    new_height = height // downscale_factor  
    new_width = width // downscale_factor  
      
    # 重新排列Tensor以匹配目标形状  
    # 首先，将Tensor展平为二维，其中每个元素对应原始图像中的一个像素块  
    input_flat = input.view(batch_size, channels, -1)  
      
    # 然后，我们需要将这个二维Tensor重新排列成三维，  
    # 其中第三维是原始的像素块（大小为downscale_factor x downscale_factor）  
    # 这需要我们先将其重塑为四维，然后再进行permute和reshape  
    # 注意：这里使用了一种巧妙的方法来重新排列，但可能不是最优的  
    # 我们首先将height和width的乘积拆分为多个像素块  
    input_reshaped = input_flat.view(batch_size, channels, new_height, downscale_factor, new_width, downscale_factor)  
      
    # 然后，我们交换维度以将像素块放在前面，并重新排列它们以匹配新的通道数  
    # 注意：permute的顺序取决于你希望如何组织像素块中的值  
    # 这里我们选择将原始的height和width维度合并到新的channels维度中  
    input_transposed = input_reshaped.permute(0, 1, 3, 5, 2, 4)  
      
    # 最后，将新的channels维度合并，并重塑为最终的二维Tensor  
    # 但注意，我们实际上需要再次reshape到三维，因为我们需要保留batch_size这一维  
    result = input_transposed.contiguous().view(batch_size, new_channels, new_height, new_width)  
      
    return result  
  
# # 示例使用  
# input_tensor = torch.randn(1, 3, 12, 16)  # 假设原始图像大小为3x4x4，downscale_factor=3不适用，这里仅为示例  
# downscale_factor = 2  # 使用一个合理的降采样因子  
# output_tensor = pixel_unshuffle_custom(input_tensor, downscale_factor)  
# print(output_tensor.shape)  # 应该输出torch.Size([1, 12, 6, 8])，但注意这只是一个形状匹配的示例，实际逻辑可能需要根据具体需求调整

# import torch.nn.functional as F

# torch_output = F.pixel_unshuffle(input_tensor,downscale_factor)
# print(torch_output.shape)
# print(output_tensor == torch_output)

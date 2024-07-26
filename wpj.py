import torch
import pdb

def load_model(single_gpu_model, saved_model_path):
    # 加载用DataParallel保存的模型状态字典
    state_dict = torch.load(saved_model_path)

    # 创建一个新的状态字典，移除'module.'前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 移除`module.`前缀
        new_state_dict[name] = v
    # pdb.set_trace()
    # 加载修改后的状态字典到单卡模型
    single_gpu_model.load_state_dict(new_state_dict)

    return single_gpu_model


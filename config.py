from model.model import create_model
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets.targets import gen_targets,prep_sample
# If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
# If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
out_shape = [164, 164] # patch shape at output of network

nr_class = 7 #Number of classses including background.
run_info = {}
run_info['mode'] = 'train'
run_info['save_path'] = 'checkpoint'
run_info['nr_epochs'] = 1000
run_info['batch_size'] = {"train": 4, "valid": 4,}
run_info['nr_class'] = nr_class
run_info['net'] = lambda: create_model(input_ch=3, nr_types=nr_class)
run_info['dir_path'] = {'train':['/path to /train'],\
'valid':['/path to /val']}
run_info['optimizer'] = optim.Adam
run_info['lr_scheduler'] = lambda x: CosineAnnealingLR(x,(4900//run_info['batch_size']['train'])*run_info['nr_epochs'])
run_info['loss'] = {"np": {"bce": 1, "dice": 1},"hv": {"mse": 1, "msge": 1},"tp": {"bce": 1, "dice": 1},}
run_info['target_info'] = {"gen": (gen_targets, {}), "viz": (prep_sample, {})}
run_info['shape_info'] = {
    "train": {"input_shape": act_shape, "mask_shape": out_shape,},
    "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
}



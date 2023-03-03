import torch

def make_batch_arg(args, num_batch):
    batch_idx_list = torch.tensor(range(len(args))).split_tensor(num_batch)
    return [[args[i] for i in batch_idx] for batch_idx in batch_idx_list]

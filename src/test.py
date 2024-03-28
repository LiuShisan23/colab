import torch
import numpy as np

att_mask = np.array([[1,0,1,0,1]])
att_mask = torch.from_numpy(att_mask)
print(att_mask.shape)
print((att_mask > 0).unsqueeze(1).repeat(1,att_mask.size(1),1).unsqueeze(1))
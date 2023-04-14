import torch
from torch import nn


class MSE(nn.Module):
    def __init__(self, cloud_index):
        super(MSE, self).__init__()
        self.cloud_index = cloud_index

    def forward(self, S2_pred, S2_true, S2_mask):
        mask = torch.ones_like(S2_mask, dtype=torch.bool)
        for index in self.cloud_index:
            mask &= (S2_mask != index)
        
        S2_true_masked = S2_true[mask]
        S2_pred_masked = S2_pred.squeeze(1)[mask]
        
        loss = torch.mean(torch.pow((S2_true_masked - S2_pred_masked), 2))
        
        return loss
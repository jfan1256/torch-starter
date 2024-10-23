import torch
import torch.nn as nn

class ModelName(nn.Module):
    def __init__(self,
                 configs,
                 ):
        super().__init__()

        # Configs
        self.configs = configs

        # TODO: ADD CODE HERE

    def forward(self, index, device):
        # TODO: ADD CODE HERE

        # Loss
        loss_ce = torch.zeros(1)

        # Return a dictionary of losses
        return {
            'loss_ce': loss_ce,
        }
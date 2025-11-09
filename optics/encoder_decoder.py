import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_generator(size=[1], a=1, seed=None):
    # Random seed vector generator
    if seed:
        torch.manual_seed(seed)

    return (torch.rand(size, device=device) - 0.5) * 2 * a

class Encoder(nn.Module):
    # MLP encoder that outputs Zernike coefficients given a random seed vector
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()

        self.act = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc4 = nn.Linear(hidden_size, output_size, bias=True)

        self.mask = torch.ones(output_size).to(device)


        self.mask[0]=0
        self.mask[1]=0

        self.relu6 = nn.ReLU6()
        
    def forward(self, x=None):

        if x is None:
            raise ValueError
            x = (torch.rand([1], device='cuda') - 0.5) * 2 * 1

        x = self.fc1(x)
        x = self.act(x) 
        
        x = self.fc2(x)
        x = self.act(x)
        
        # x = self.fc3(x)
        # x = self.act(x)

        x = self.fc4(x)
        
        # x = self.relu(x)
        if len(self.mask) == 350:
            x = (torch.sigmoid((x)) - 0.5) * 2

        else:
            raise ValueError

        return x * self.mask

def zernike_generator_16_350():
    return Encoder(16, 512, 350)
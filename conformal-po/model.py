import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import tqdm

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super(MLP, self).__init__()
        order_dict = []
        for i in range(len(hidden_sizes) - 1):
            order_dict.append(('Linear Layer {}'.format(i), nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
            if i < len(hidden_sizes) - 2:
                order_dict.append(('BatchNorm Layer {}'.format(i), nn.BatchNorm1d(hidden_sizes[i+1])))
                order_dict.append(('ReLU Layer {}'.format(i), nn.ReLU()))
        self.mlp = nn.Sequential(OrderedDict(order_dict))
    
    def forward(self, x):
        return self.mlp(x)
    
class Encoder(nn.Module):
    def __init__(self, feature_size, hidden_sizes, latent_size, cond_size = 0):
        super(Encoder, self).__init__()
        self.encoder = MLP([feature_size + cond_size] + hidden_sizes)
        self.cond_size = cond_size
        self.latent_size = latent_size
        self.means = MLP([hidden_sizes[-1]] + [latent_size])
        self.log_var = MLP([hidden_sizes[-1]] + [latent_size])
    
    def forward(self, input, cond = None):
        if self.cond_size > 0:
            input = torch.cat([input, cond], dim = 1)
        new_input = self.encoder(input)
        return self.means(new_input), self.log_var(new_input)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
            
class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_sizes, feature_size, cond_size = 0):
        super(Decoder, self).__init__()
        self.cond_size = cond_size
        self.latent_size = latent_size
        self.decoder = MLP([latent_size + cond_size] + hidden_sizes + [feature_size])
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, z, cond = None):
        # print(z.shape, cond.shape)
        if z.shape[0] != cond.shape[0]:
            cond = cond.repeat(z.shape[0], 1)
        if self.cond_size > 0:
            z = torch.cat([z, cond], dim = 1)
        return self.decoder(z)




class CVAE(nn.Module):
    def __init__(self, feature_size, encoder_sizes, latent_size, decoder_sizes, cond_size = 0):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.cond_size = cond_size
        self.latent_size = latent_size
        self.encoder = Encoder(feature_size, encoder_sizes, latent_size, cond_size)
        self.decoder = Decoder(latent_size, decoder_sizes, feature_size, cond_size)
    
    def forward(self, input, cond = None):
        means, log_var = self.encoder(input, cond)
        z = self.encoder.reparameterize(means, log_var)
        return self.decoder(z, cond), means, log_var
    
    def generate(self, cond, n):
        z = torch.randn(n, self.latent_size).to(cond.device)
        return self.decoder(z, cond)



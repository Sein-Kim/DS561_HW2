from torch import nn
import torch
import torch.functional as F
import torch.nn.functional as F2
import time
import math
import numpy as np

class Encoder(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(Encoder, self).__init__()
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, kernel_size), padding=dilation, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)

        self.gate = nn.Sigmoid()
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x): # x: [batch_size, seq_len, embed_size]
        # x = x.unsqueeze(1)
        out =  self.conv1(x.unsqueeze(1))
        out = F2.relu(out)

        return out

    def conv_pad(self, x, dilation):
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, residual = True):
        super(ResidualBlock, self).__init__()
        self.out_channel = out_channel
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, kernel_size), padding=dilation, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, kernel_size), padding=dilation*2, dilation=dilation*2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.gate = nn.Sigmoid()
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x): # x: [batch_size, seq_len, embed_size]
        # x = x.unsqueeze(1)
        out =  self.conv1(x)
        out = F2.relu(out)

        out = self.conv2(out)
        out = F2.relu(out)

        if self.residual:
            out = out + x

        return out

    def conv_pad(self, x, dilation):
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class Residual_CNN(nn.Module):

    def __init__(self, model_para):
        super(Residual_CNN, self).__init__()
        self.model_para = model_para
        self.embed_size = model_para['dilated_channels']
        self.residual = model_para['residual']
        self.dilations = model_para['dilations']
        self.residual_channels = model_para['dilated_channels']
        self.kernel_size = model_para['kernel_size']
        
        self.encoder = Encoder(1, self.residual_channels, kernel_size=self.kernel_size, dilation=1)
        rb = [ResidualBlock(self.residual_channels, self.residual_channels, kernel_size=self.kernel_size,
                            dilation=dilation, residual = self.residual) for dilation in self.dilations]
        self.residual_blocks = nn.Sequential(*rb)

        self.final_layer = nn.Linear(self.residual_channels, 1)

        self.final_layer.weight.data.normal_(0.0, 0.01)  # initializer
        self.final_layer.bias.data.fill_(0.1)

    def forward(self, x,onecall=False): # inputs: [batch_size, seq_len]
        x = self.encoder(x)
        dilate_outputs = self.residual_blocks(x)
        dilate_outputs = dilate_outputs.mean(axis=3)
        out = self.final_layer(dilate_outputs.view(-1,self.residual_channels))
        # if onecall:
        #     hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels) # [batch_size, embed_size]
        # else:
        #     hidden = dilate_outputs.view(-1, self.residual_channels) # [batch_size*seq_len, embed_size] 
            

        return out

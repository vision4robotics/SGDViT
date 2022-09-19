
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size,mask = None):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    if mask is None:
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size) #B, H // window_size, W // window_size, C, window_size,  window_size
    else:
#####
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, -1, C, window_size, window_size)
    return windows

def maskwindow_partition(x, window_size,mask = None):
    #B, N_*num, C, window_size, window_size
    
    B, N_, C, H, W = x.shape
    
    x = x.view(B, N_, C, H // window_size, window_size, W // window_size, window_size)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(B, -1, C, window_size, window_size) #B, N_*num, C, window_size, window_size

    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x

def maskwindow_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B, N, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B, N, C, _, _ = windows.shape
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x


def position(window_size,num_heads):
    # define a parameter table of relative position bias
    relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

    # get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    trunc_normal_(relative_position_bias_table, std=.02)

    return relative_position_bias_table, relative_position_index


# mask = torch.rand(2,1,28,28)
# x = torch.rand(2,256,28,28)
# _, C, _, _ =x.shape
# size = 4
# layer1 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
# layer2 = nn.BatchNorm2d(256)
# layer3 = nn.ReLU(inplace=True)
# mask = F.gumbel_softmax(mask, hard=True)
# x_windows = window_partition(x, size, mask=True)                   #B, num_windows, C, window_size, window_size
# mask_windows = window_partition(mask, size, mask=True)             #B, num_windows, 1, window_size, window_size
# mask_windows = mask_windows.flatten(3)                             #B, num_windows, 1,  window_size*window_size
# mask_windows = torch.norm(mask_windows, p=1, dim=3)
# mask_windows, indices = torch.sort(mask_windows, dim = 1)
# print(indices.shape)
# B, K, _ = indices.shape
# keep_node = 17
# indices = torch.split(indices, [keep_node,K - keep_node], dim=1)  
# index = torch.squeeze(indices[0])
# index_ = torch.squeeze(indices[1])
# print(index.shape)
# x_windows_s = x_windows[torch.arange(B)[:,None],index,:]
# print(x_windows.shape)
# print(x_windows_s.shape)
# x_windows_b = x_windows[torch.arange(B)[:,None],index_,:]
# x_windows_s = maskwindow_partition(x_windows_s, size//2)
# print(x_windows_s.shape)
# print(x_windows_b.shape)
# x_windows_b = layer1(x_windows_b.view(-1, C, size, size))
# x_windows_b = layer2(x_windows_b)
# x_windows_b = layer3(x_windows_b)
# x_windows_b=x_windows_b.view(B, -1, C, size//2, size//2)
# print(x_windows_b.shape)
# x = torch.cat([x_windows_b,x_windows_s], dim=1)


# x = maskwindow_reverse(x,size//2,20,20)
# print(x.shape)




import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.models.utils.windows import window_partition,maskwindow_reverse,maskwindow_partition 

class MaskEmbed(nn.Module):
    """
    Input: 1, 256, 16, 16
    Output:1, 256, 14, 14
    """
    def __init__(self, window_size=4, dim=256, embed_dim=256, keep_node=11):
        super().__init__()
        self.size = window_size
        self.keep_node = keep_node
        self.layer1 = nn.Conv2d(dim, embed_dim, kernel_size=self.size//2, stride=self.size//2)
        self.layer2 = nn.BatchNorm2d(embed_dim)
        self.layer3 = nn.ReLU(inplace=True)

    def forward(self, x, mask, size=4):
        _, C, _, _ = x.shape
        mask = F.gumbel_softmax(mask, hard=True)
        real = mask
        x_windows = window_partition(x, size, mask=True)                   #B, num_windows, C, window_size, window_size
        mask_windows = window_partition(mask, size, mask=True)             #B, num_windows, 1, window_size, window_size
        mask_windows = mask_windows.flatten(3)                             #B, num_windows, 1,  window_size*window_size
        mask_windows = torch.norm(mask_windows, p=1, dim=3)
        mask_windows, indices = torch.sort(mask_windows, dim = 1)
        B, K, _  = indices.shape
        indices = torch.split(indices, [self.keep_node,K - self.keep_node], dim=1)  
        index = torch.squeeze(indices[0])
        index_ = torch.squeeze(indices[1])
        x_windows_s = x_windows[torch.arange(B)[:,None],index,:]           #B, num_windows_, C, window_size, window_size
        x_windows_b = x_windows[torch.arange(B)[:,None],index_,:]
        x_windows_s = maskwindow_partition(x_windows_s, size//2)
        x_windows_b = self.layer1(x_windows_b.view(-1, C, size, size))
        x_windows_b = self.layer2(x_windows_b)
        x_windows_b = self.layer3(x_windows_b)
        x_windows_b=x_windows_b.view(B, -1, C, size//2, size//2)
        x = torch.cat([x_windows_b,x_windows_s], dim=1)
        # print(x.shape)
        x = maskwindow_reverse(x,size//2,14,14)

        return x#
    
    if __name__ == '__main__':
    x = torch.rand(6,256,16,16)
    z = torch.rand(6,1,16,16)
    layer =MaskEmbed()
    x = layer(x,z)
    print(x.shape)
    
    

     

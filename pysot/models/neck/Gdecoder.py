
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class Decoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(activation)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Tensor, saliency:Tensor, template: Tensor, pos_enc1: Optional[Tensor] = None,
                     pos_dec1: Optional[Tensor] = None, pos_enc2: Optional[Tensor] = None) -> Tensor:

        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, pos_dec1), self.with_pos_embed(saliency, pos_enc1), saliency)[0]
        tgt2 = tgt + self.dropout1(tgt2)
        tgt2 = self.norm1(tgt2)
        tgt2 = self.multihead_attn(tgt2, self.with_pos_embed(template, pos_enc2), template)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Gdecoder(nn.Module):
    def __init__(self,channel=256):
        super().__init__()
        self.layer=Decoder(channel,8)
        self.row_embed = nn.Embedding(50, channel//2)
        self.col_embed = nn.Embedding(50, channel//2)
    def pos_embedding(self,x):
        h, w = x.shape[-2:]
        #print(h,w)
        i = torch.arange(w).cuda()
        j = torch.arange(h).cuda()
        # i = torch.arange(w)
        # j = torch.arange(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        pos=pos.flatten(2).permute(2,0,1)
        # print(pos.shape)
        return pos

    def forward(self,search,template,saliency):
        B, C, H, W = search.shape
        pos_1=self.pos_embedding(saliency)
        pos_2=self.pos_embedding(search)
        pos_3=self.pos_embedding(template)
        search=search.flatten(2).permute(2,0,1)
        template=template.flatten(2).permute(2,0,1)
        saliency=saliency.flatten(2).permute(2,0,1)
        x=self.layer(tgt=search,saliency=saliency,template=template, pos_dec1=pos_2, 
                        pos_enc1=pos_1, pos_enc2=pos_3)
        x = x.permute(1,2,0).view(B, C, H, W)
        
        return x

        

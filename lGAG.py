import torch
import torch.nn as nn
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//2 , kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//2, channel, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            SEBlock(F_int),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU()




    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x*psi

# class LGAG(nn.Module):
#     def __init__(self, F_g, F_l, F_int, num_heads=2, kernel_size=3, groups=1, activation='relu'):
#         super(LGAG, self).__init__()
#
#         # if kernel_size == 1:
#         #     groups = 1
#
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
#                       bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
#                       bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=F_int, num_heads=num_heads)
#
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#
#         self.activation = nn.ReLU(inplace=True)  # »òÕßÊ¹ÓÃÆäËûŒ€»îº¯Êý
#
#
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#
#
#         g1_flat = g1.flatten(2).permute(2, 0, 1)  # (Batch, Channel, Height*Width) -> (Height*Width, Batch, Channel)
#         x1_flat = x1.flatten(2).permute(2, 0, 1)
#
#
#         attn_output, _ = self.multihead_attn(g1_flat, x1_flat, x1_flat)
#
#
#         attn_output = attn_output.permute(1, 2, 0).view_as(
#             g1)  # (Height*Width, Batch, Channel) -> (Batch, Channel, Height, Width)
#
#         psi = self.activation(attn_output)
#         psi = self.psi(psi)
#
#         return x * psi
# a=torch.rand(1,32,224,224)
# b=torch.rand(1,32,224,224)
# w=LGAG(F_g=32,F_l=32,F_int=32)
# print(w(a,b).shape)
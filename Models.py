import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

def gen_grid_up(up_ratio, grid_size=0.2):
    sqrted = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break

    grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
    grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)

    x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
    grid = torch.stack([x, y], dim=-1).view(-1, 2).transpose(0, 1).contiguous()
    return grid

class ModelEncoder2(nn.Module):
    def __init__(self, input_points, emb_dim=20, total_points=None, scale=None):
        super(ModelEncoder2, self).__init__()
        self.emb_dim = emb_dim
        assert not (total_points and scale), "Please provide either total_points or scale"
        if not total_points and not scale:
          scale = 2
        if total_points:
          scale = total_points/input_points
        self.scale = scale
        self.input_points = input_points
        self.conv1 = nn.Conv1d(3, emb_dim, 1)
        self.conv2 = nn.Conv1d(emb_dim, emb_dim, 1)
        self.ff = nn.Flatten()
        self.l1 = nn.Linear(emb_dim*input_points, round(scale*emb_dim*input_points))
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=input_points,
            num_blocks=7,
            embedding_dim=emb_dim,
            slstm_at=[1],
        )
        self.cfg2 = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=emb_dim,
            num_blocks=7,
            embedding_dim=input_points*scale,
            slstm_at=[1],
        )
        self.xlstm_stack = xLSTMBlockStack(self.cfg)
        self.xlstm_stack2 = xLSTMBlockStack(self.cfg2)
        self.conv_lst1 = nn.ModuleList([nn.Conv1d(emb_dim, emb_dim, 1), nn.ReLU(), nn.Conv1d(emb_dim, emb_dim, 1), nn.ReLU(), nn.Conv1d(emb_dim, emb_dim, 1), nn.RReLU()])
        self.conv_lst2 = nn.ModuleList([nn.Conv1d(emb_dim, emb_dim, 1), nn.ReLU(), nn.Conv1d(emb_dim, emb_dim, 1), nn.ReLU(), nn.Conv1d(emb_dim, emb_dim, 1)])
        self.conv_lst3 = nn.ModuleList([nn.Conv1d(emb_dim, emb_dim, 1), nn.ReLU(), nn.Conv1d(emb_dim, emb_dim, 1), nn.ReLU(), nn.Conv1d(emb_dim, emb_dim, 1), nn.GELU()])
        self.conv3 = nn.Conv1d(emb_dim, 3, 1)
    def forward(self, x):
        x = x.transpose(2,1).contiguous()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(2, 1).contiguous()
        x = self.xlstm_stack(x).transpose(2, 1).contiguous()
        y = x.clone()
        z = x.clone()
        for i in self.conv_lst1: x = i(x)
        for i in self.conv_lst2: y = i(y)
        for j in self.conv_lst3: z = i(z)
        x = x + y + z
        # x = self.ff(x)
        # x = F.relu(self.l1(x)).view(-1, self.emb_dim, self.scale*self.input_points)
        # x = self.xlstm_stack2(x)
        return self.conv3(x).transpose(1,2)


class XL_encoder(nn.Module):
    def __init__(self, num_points=2048, output_size=1024):
        super(XL_encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=num_points,
            num_blocks=4,
            embedding_dim=128,
            slstm_at=[1],
        )
        cfg2 = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=num_points,
            num_blocks=4,
            embedding_dim=512,
            slstm_at=[1],

        )

        self.xlstm_point_stack = xLSTMBlockStack(cfg)
        self.xlstm_point_stack2 = xLSTMBlockStack(cfg2)
    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = x.transpose(1,2).contiguous()
        x = self.xlstm_point_stack(x).transpose(1,2).contiguous()
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = x.transpose(1,2).contiguous()
        x = self.xlstm_point_stack2(x).transpose(1,2).contiguous()
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)


class PCN_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, cat_feature_num):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()
        self.conv1 = nn.Conv1d(cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()

        point_feat = (
            (coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                3)).transpose(1,
                                                                                                              2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                      3)).transpose(1,
                                                                                                                    2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine


class Model(nn.Module):
    def __init__(self, num_points, num_coarse=1024):
        super(Model, self).__init__()

        self.num_coarse = num_coarse
        self.num_points = num_points
        self.scale = self.num_points // num_coarse
        self.cat_feature_num = 2 + 3 + 1024

        self.encoder = XL_encoder()
        self.decoder = PCN_decoder(num_coarse, self.num_points, self.scale, self.cat_feature_num)

    def forward(self, x):
        feat = self.encoder(x)
        out1, out2 = self.decoder(feat)
        out1 = out1.transpose(1, 2).contiguous()
        out2 = out2.transpose(1, 2).contiguous()
        return out1, out2



import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLoss(nn.Module):
    """
    Layer-wise attention weighted feature alignment loss
    """

    def __init__(self, feat_dim=192, proj_dim=None):
        super().__init__()

        if proj_dim is None:
            proj_dim = feat_dim

        self.t_k_project = nn.Linear(feat_dim, proj_dim, bias=False)
        self.t_v_project = nn.Linear(feat_dim, proj_dim, bias=False)
        self.s_q_project = nn.Linear(feat_dim, proj_dim, bias=False)
        self.s_v_project = nn.Linear(feat_dim, proj_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, t_feat, s_feat, teacher_index):
        B, C, H, W = t_feat.shape
        t_feat = t_feat.view(B, C, -1).permute(0, 2, 1)
        B, C, H, W = s_feat.shape
        s_feat = s_feat.view(B, C, -1).permute(0, 2, 1)[teacher_index]
        t_k = self.relu(self.t_k_project(t_feat))
        t_v = self.t_v_project(t_feat)
        s_q = self.relu(self.s_q_project(s_feat))
        s_v = self.s_v_project(s_feat)

        attn = torch.matmul(s_q, t_k.transpose(-1, -2))
        attn = attn / (t_k.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        t_vv = (attn @ t_v)
        featureLoss = torch.norm(t_vv - s_v, dim=-1).mean()
        return featureLoss / len(t_feat)

class DomainClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.net = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.net(x)
    
class grlLoss(nn.Module):
    def __init__(self, feat_dim=192, proj_dim=192, grl=None):
        super().__init__()

        if proj_dim is None:
            proj_dim = feat_dim
        self.k_project = nn.Linear(feat_dim, proj_dim, bias=False)
        self.v_project = nn.Linear(feat_dim, proj_dim, bias=False)
        self.q_project = nn.Linear(feat_dim, proj_dim, bias=False)
        self.grl = grl
        self.relu = nn.ReLU(inplace=True)
        self.D_o = DomainClassifier(proj_dim)
    def forward(self, s_feat, teacher_index):
        B, C, H, W = s_feat.shape
        s_feat = s_feat.view(B, C, -1).permute(0, 2, 1)
        s_k = self.relu(self.k_project(s_feat))
        s_q = self.relu(self.q_project(s_feat))
        s_v = self.v_project(s_feat)

        attn = torch.matmul(s_q, s_k.transpose(-1, -2))
        attn = attn / (s_k.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        s_vv = (attn @ s_v)
        feat = s_vv.mean(dim=1)
        feat = F.normalize(feat, dim=-1)
        
        y = torch.zeros(feat.shape[0], dtype=torch.long).to(feat.device)
        y[teacher_index] = 1.0  
        y[~teacher_index] = 0.0
        logits_o = self.D_o(self.grl(feat))   
        loss_adv = F.cross_entropy(logits_o, y.to(feat.device))
        return loss_adv
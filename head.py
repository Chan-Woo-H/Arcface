import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self,in_feature=128, out_feature=10575, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sine_m = math.cos(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.uniform_(self.weight,0.0,0.5)
        
    def forward(self, cosine, label):

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda:3')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
        
#        index = torch.where(label != -1)[0]
#        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
#        m_hot.scatter_(1, label[index, None], self.m)
#        cosine.acos_()
#        cosine[index] += m_hot
#        cosine.cos_().mul_(self.s)
#        return cosine

class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret

def load_L(name):
    if name == "arc":
        return ArcFace()
    if name == "cos":
        return CosFace() 
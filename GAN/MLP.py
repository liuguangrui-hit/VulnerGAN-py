from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self,layers=[],resume=None):
        super(MLP,self).__init__()
        self.make_layers(layers)
        if resume!=None:
            self.model.load_state_dict(torch.load(resume))

    def make_layers(self,layers):
        model_list=[]
        for layer in layers:
            if layer[0]=="s":
                model_list.append(nn.Sigmoid())
            elif layer[0]=="fc":
                model_list.append(nn.Linear(layer[1],layer[2]))
            elif layer[0]=="relu":
                model_list.append(nn.ReLU())
            elif layer[0]=="lrelu":
                model_list.append(nn.LeakyReLU())

        self.model=nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)
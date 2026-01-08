import torch
import math
from torch import nn
from function_transformer_attention import SpGraphTransAttentionLayer
from base_classes import ODEblock
import os
import numpy as np
import random
from utils import get_rw_adj
from torchfde import fdeint_general, fdeint, fdeint1, fdeint2

from torchfde.learnable_solver_new import LearnbleFDEINT

class AttODEblock_FRAC(ODEblock):          # NIPS2025 AttentionKernel
  def __init__(self, odefunc,  opt, data,  device, t=torch.tensor([0, 1])):
    super(AttODEblock_FRAC, self).__init__(odefunc,  opt, data, device, t)

    self.odefunc = odefunc( opt['hidden_dim'], opt['hidden_dim'], opt, data, device)
    # self.odefunc.edge_index, self.odefunc.edge_weight = data.edge_index, edge_weight=data.edge_attr
    edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                         fill_value=opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)


    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:

      from torchdiffeq import odeint
    self.train_integrator = odeint
    self.test_integrator = odeint

    self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], opt['hidden_dim'], opt,device, edge_weights=self.odefunc.edge_weight).to(device)
    self.device = device
    self.opt = opt
    self.solver1 = LearnbleFDEINT(state_dim=opt['hidden_dim']).to(device)

  
  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention

  def forward(self, x):

    t = self.t.type_as(x)

    self.odefunc.attention_weights = self.get_attention_weights(x)



    func = self.odefunc
    state = x


    alpha = torch.tensor(self.opt['alpha_ode'])

    z = self.solver1(func, state, alpha, t=self.opt['time'], step_size=self.opt['step_size'], method=self.opt['method'])

    return z


  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"

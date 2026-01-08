from base_classes import ODEblock
import torch
from torch import nn
from utils import get_rw_adj, gcn_norm_fill_val
from torchfde import fdeint, fdeint1, fdeint2
from torchfde.learnable_solver_new import LearnbleFDEINT
from torchfde.learnable_solver_new import LearnbleFDEINT_variable

class ConstantODEblock_FRAC(ODEblock):
  def __init__(self, odefunc, opt, data,  device, t=torch.tensor([0, 1])):
    super(ConstantODEblock_FRAC, self).__init__(odefunc,  opt, data,   device, t)

    self.odefunc = odefunc(opt['hidden_dim'], opt['hidden_dim'], opt, data, device)
    if opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                   fill_value=opt['self_loop_weight'],
                                                                   num_nodes=data.num_nodes,
                                                                   dtype=data.x.dtype)
    else:
      edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
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
    # self.set_tol()
    self.device = device
    self.opt = opt
    # self.solver1 = LearnbleFDEINT(state_dim=opt['hidden_dim']).to(device)
    self.solver1 = LearnbleFDEINT(hidden_dim=opt['hidden_dim'], state_dim=opt['hidden_dim']).to(device)

  def forward(self, x):
    t = self.t.type_as(x)

    integrator = self.train_integrator if self.training else self.test_integrator
    

    func = self.odefunc
    state = x


    alpha = torch.tensor(self.opt['alpha_ode'])

    if alpha > 1:
        raise ValueError("alpha_ode must be in (0,1)")


    z = self.solver1(func, state, alpha, t=self.opt['time'], step_size=self.opt['step_size'], method=self.opt['method'])
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"



  



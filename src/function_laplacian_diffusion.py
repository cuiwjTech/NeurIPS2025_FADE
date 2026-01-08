import torch
from torch import nn
import torch_sparse

from collections import deque
from base_classes import ODEFunc
from utils import MaxNFEException
from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
from torch_geometric.utils import get_laplacian

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention','att_frac']:  # adj is a multihead attention
      # print("cuiwenjun20", )  #go 
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      # print("cuiwenjun21", )  #not go 
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      # print("cuiwenjun21", )  # constant_frac go
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    self.nfe += 1
    # print("x shape: ", x.shape)
    # print("edge_index shape: ", self.edge_index.shape)
    ax = self.sparse_multiply(x)

    # ax = torch.cat([x, ax], axis=1)
    # ax = self.lin2(ax)

    if not self.opt['no_alpha_sigmoid']:
      # print("cuiwenjun21", )    #go
      alpha = torch.sigmoid(self.alpha_train)
    else:
      # print("cuiwenjun22", )
      alpha = self.alpha_train
    # f = ax -x
    f = alpha * (ax - x)
    if self.opt['add_source']:
      # print("cuiwenjun23", )   #go
      # print("cuiwenjun23", self.x0)
      f = f + self.beta_train * self.x0
    return f
  


class LaplacianODEFunc1(ODEFunc):          # 含有delay
    def __init__(self, in_features, out_features, opt, data, device):
      super(LaplacianODEFunc1, self).__init__(opt, data, device)

      self.in_features = in_features
      self.out_features = out_features
      self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
      self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
      self.alpha_sc = nn.Parameter(torch.ones(1))
      self.beta_sc = nn.Parameter(torch.ones(1))
      self.c1 = nn.Parameter(torch.ones(1))
      self.d1 = nn.Parameter(torch.ones(1))

      # self.num_previous_states = num_previous_states  # 前一状态的数量
      # self.previous_states = deque(maxlen=num_previous_states)  # 用队列存储前一状态
      self.previous_weights = nn.Parameter(torch.zeros(5))  # 前一状态的可学习权重

    
    def sparse_multiply(self, x):
      if self.opt['block'] in ['attention','att_frac']:  # adj is a multihead attention
        # print("cuiwenjun20", )  #go block att_frac 
        mean_attention = self.attention_weights.mean(dim=1)
        ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
      elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
        # print("cuiwenjun21", )  #not go 
        ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
      else:  # adj is a torch sparse matrix
        # print("cuiwenjun22", )  #go block constant_frac
        ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
      return ax

    # def add_state(new_state):
    #     previous_states = []
    #     # 检查列表长度是否达到3
    #     if len(previous_states) == 3:
    #     # 删除列表中的第0个元素
    #         previous_states.pop(0) 
    #     # 添加新的状态到列表末尾
    #     previous_states.append(new_state)
    #     return previous_states
    
    def forward(self, t, x):  # ODE 求解器需要 t 参数
      self.nfe += 1
      ax = self.sparse_multiply(x)

      if not self.opt['no_alpha_sigmoid']:
        alpha = torch.sigmoid(self.alpha_train)
      else:
        alpha = self.alpha_train
        
        # 计算当前输出
      # print("cuiwenjun1", )
      f = alpha * (ax - x)

      if self.opt['add_source']:
        # print("cuiwenjun2", )
        f = f + self.beta_train * self.x0
      
      previous_states = []
        # 检查列表长度是否达到3
      if len(previous_states) == 5:
        # 删除列表中的第0个元素
        previous_states.pop(0) 
        # 添加新的状态到列表末尾
      previous_states.append(f)
        
      # for i, prev_state in enumerate(self.previous_states):
      #     # f = f + self.previous_weights[i] * prev_state
      #   f = f + self.previous_weights[i] * (prev_state - x)

      weighted_sum = 0
      for i in range(len(previous_states)):
        # f = f + self.previous_weights[i] * (previous_states[i] - f)
        # print("cuiwenjun1", )
        # f = f + self.previous_weights[i] * previous_states[i]
        # weighted_sum += self.previous_weights[i] * (previous_states[i] - f)
        # weighted_sum += self.previous_weights[i] * (previous_states[i]-0)
        weighted_sum = weighted_sum + self.previous_weights[i] * (previous_states[i]-0)
        # weighted_sum += self.previous_weights[i] * (previous_states[i] - f)
      # f = f + weighted_sum
      f = self.c1 * f + self.d1 * weighted_sum

    
      
      # if self.opt['add_source']:
      #   f = f + self.beta_train * self.x0

        # 更新前一状态缓冲区
      # self.previous_states.append(x.detach().clone())
      # print("cuiwenjun24", )

      #  # 添加梯度裁剪
      # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

      #   # 使用 torch.clamp 避免出现 NaN 值
      # f = torch.clamp(f, min=-1e6, max=1e6)


      return f


class LaplacianODEFunc2(ODEFunc):          # 含有delay
    def __init__(self, in_features, out_features, opt, data, device):
      super(LaplacianODEFunc2, self).__init__(opt, data, device)

      self.in_features = in_features
      self.out_features = out_features
      self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
      self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
      self.alpha_sc = nn.Parameter(torch.ones(1))
      self.beta_sc = nn.Parameter(torch.ones(1))
      self.c1 = nn.Parameter(torch.ones(1))
      self.d1 = nn.Parameter(torch.ones(1))

      # self.num_previous_states = num_previous_states  # 前一状态的数量
      # self.previous_states = deque(maxlen=num_previous_states)  # 用队列存储前一状态
      # self.previous_weights = nn.Parameter(torch.zeros(6))  # 前一状态的可学习权重
      self.fhistory = []
      self.memory = opt.get('memory', 5)
      self.memory_weights = nn.Parameter(torch.ones(self.memory) / self.memory)

    
    def sparse_multiply(self, x):
      if self.opt['block'] in ['attention','att_frac']:  # adj is a multihead attention
        # print("cuiwenjun20", )  #go block att_frac 
        mean_attention = self.attention_weights.mean(dim=1)
        ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
      elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
        # print("cuiwenjun21", )  #not go 
        ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
      else:  # adj is a torch sparse matrix
        # print("cuiwenjun22", )  #go block constant_frac
        ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
      return ax
    
    def forward(self, t, x):  # ODE 求解器需要 t 参数
      self.nfe += 1
      ax = self.sparse_multiply(x)

      if not self.opt['no_alpha_sigmoid']:
        alpha = torch.sigmoid(self.alpha_train)
      else:
        alpha = self.alpha_train
        
        # 计算当前输出
      # print("cuiwenjun1", )
      f = alpha * (ax - x)

      if self.opt['add_source']:
        # print("cuiwenjun2", )
        f = f + self.beta_train * self.x0
      
      self.fhistory.append(f)
      if len(self.fhistory) > self.memory:
        self.fhistory.pop(0)

    # 计算加权求和的f
      memory_len = len(self.fhistory)
      weights = torch.softmax(self.memory_weights[:memory_len], dim=0)
      weighted_f = torch.sum(torch.stack([weights[i] * self.fhistory[i] for i in range(memory_len)]), dim=0)


      return weighted_f


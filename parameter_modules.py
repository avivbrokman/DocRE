#%% libraries
import torch
from torch.nn import Module, LazyLinear
from torch.nn.functional import mish, max_pool2d, sigmoid

#%% intuitive max pooling
def max_pool(x):
    dim = len(x.size())
    
    if dim == 2:
        x = x.unsqueeze(0)
        
    kernel_size = (x.size(-2), 1)
    x = max_pool2d(x, kernel_size)
    
    if dim == 2:
        x = x.squeeze(0)
        
    return x

#%% logistic regression (typical prediction layer)
# class LogisticRegression(Module):
#     def __init__(self, num_classes):
#         super().__init__()
        
#         self.linear = LazyLinear(num_classes)
        
#     def forward(self, x):
#         x = self.linear(x)
#         x = softmax(x, -1)
        
#         return x
    
#%% MLP2
class MLP2(Module):
    def __init__(self, intermediate_dim, output_dim, activation = mish, final_activation = False):
        super().__init__()
        
        self.linear1 = LazyLinear(intermediate_dim)
        self.linear2 = LazyLinear(output_dim)
        self.activation = activation
        self.final_activation = final_activation

        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        if self.final_activation:
            x = self.activation(x)
        
        return x
    
#%% Identity
class Identity(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

#%% MLP2 Pooler
class MLP2Pooler(Module):
    def __init__(self, intermediate_dim, output_dim):
        super().__init__()

        self.mlp = MLP2(intermediate_dim, output_dim, final_activation = True)
        
    def forward(self, x):

        x = self.mlp(x)
        x = max_pool(x)
        
        return x
    
#%% Attention Pooler
class AttentionPooler(Module):
    def __init__(self, intermediate_dim, output_dim):
        super().__init__()

        self.mlp = MLP2(intermediate_dim, output_dim, final_activation = True)
        
    def forward(self, x):

        x = self.mlp(x)
        x = max_pool(x)
        
        return x

#%% Lazy Square Linear
class LazySquareLinear(LazyLinear):
    
    def initialize_parameters(self, input) -> None:  
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.in_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.in_features,))
                self.reset_parameters()  

#%% Gate
class Gate(Module):
    def __init__(self, dim_out):
        super().__init__()
        self.linear_focal = LazySquareLinear(bias = True)
        self.linear_extra = LazyLinear(0, bias = False)
        
    def forward(self, focal, extra):
        if not self.linear_extra.in_features:
            x_temp = self.linear_focal(focal)
            self.linear_extra.out_features = self.linear_focal.out_features
        gate = self.linear1(focal) + self.linear2(extra)
        gate = sigmoid(gate)
        return gate * focal

    

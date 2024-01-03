#%% libraries
from torch.nn import Module, LazyLinear
from torch.nn.functional import mish
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
    
#%% 




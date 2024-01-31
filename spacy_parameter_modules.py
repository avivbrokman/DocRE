#%% libraries
import torch
from torch.nn import Module, LazyLinear, Embedding, Linear
from torch.nn.functional import mish, max_pool2d, sigmoid
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

from spacy_data_classes import Span, SpanGroup

#%% Enhanced Module
class EnhancedModule(Module):
    
    def _device(self):
        return next(self.parameters()).device

#%% intuitive max pooling
def max_pool(x):
    dim = len(x.size())
    
    if dim == 2:
        x = x.unsqueeze(0)
        
    kernel_size = (x.size(-2), 1)
    x = max_pool2d(x, kernel_size)
    
    if dim == 2:
        x = x.squeeze(0,1)
        
    return x

#%% logistic regression (typical prediction layer)
# class LogisticRegression(EnhancedModule):
#     def __init__(self, num_classes):
#         super().__init__()
        
#         self.linear = LazyLinear(num_classes)
        
#     def forward(self, x):
#         x = self.linear(x)
#         x = softmax(x, -1)
        
#         return x

#%% MLP1
class MLP1(EnhancedModule):
    def __init__(self, output_dim, activation = mish):
        super().__init__()

        self.linear = LazyLinear(output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

#%% MLP2
class MLP2(EnhancedModule):
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
        if self.final_activation is True:
            x = self.activation(x)
        elif self.final_activation:
            x = self.final_activation(x)
        
        return x
    
#%% Identity
class Identity(EnhancedModule):

    def forward(self, x):
        return x

#%% MLP2 Pooler
class MLP2Pooler(EnhancedModule):
    def __init__(self, intermediate_dim, output_dim):
        super().__init__()

        self.mlp = MLP2(intermediate_dim, output_dim, final_activation = True)
        
    def forward(self, x):

        x = self.mlp(x)
        x = max_pool(x)
        
        return x
    
#%% Attention Pooler
class AttentionPooler(EnhancedModule):
    def __init__(self, intermediate_dim, output_dim):
        super().__init__()

        self.mlp = MLP2(intermediate_dim, output_dim, final_activation = True)
        
    def forward(self, x):

        x = self.mlp(x)
        x = max_pool(x)
        
        return x

#%% Lazy Square Linear
# class LazySquareLinear(LazyLinear):
    
#     def initialize_parameters(self, input) -> None:  
#         if self.has_uninitialized_params():
#             with torch.no_grad():
#                 self.in_features = input.shape[-1]
#                 self.weight.materialize((self.in_features, self.in_features))
#                 if self.bias is not None:
#                     self.bias.materialize((self.in_features,))
#                 self.reset_parameters()  

class LazySquareLinear(LazyModuleMixin, Linear):

    cls_to_become = Linear  # type: ignore[assignment]
    weight: UninitializedParameter
    bias: UninitializedParameter  # type: ignore[assignment]

    def __init__(self, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        super().__init__(0, 0, False)
        self.weight = UninitializedParameter(**factory_kwargs)
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.in_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.in_features,))
                self.reset_parameters()

#%% Gate
class Gate(EnhancedModule):
    def __init__(self):
        super().__init__()
        self.linear_focal = LazySquareLinear(bias = True)
    
    # def _device(self):
    #     return next(self.linear_focal.parameters()).device

    def forward(self, focal, extra):
        
        focal_transform = self.linear_focal(focal)
        if not hasattr(self, 'linear_extra'):
            self.linear_extra = LazyLinear(self.linear_focal.weight.shape[1], bias = False, device = self._device())
        extra_transform = self.linear_extra(extra)
        gate = focal_transform + extra_transform

        gate = sigmoid(gate)
        return gate * focal

#%% EnhancedEmbedding
    
class EnhancedEmbedding(Embedding):

    def _device(self):
        return next(self.parameters()).device
    
    def _tensorify(self, list_,):
        return torch.tensor(list_, dtype = torch.int32, device = self._device())
    
    def _process(self, items):
        return items
    
    def forward(self, x):
        x = self._process(x)
        x = self._tensorify(x)
        return Embedding.forward(self, x)
    
#%% Embeddings
class LengthEmbedding(EnhancedEmbedding):
    def _process(self, spans):
        return [len(el) for el in spans]

class LengthDiffEmbedding(EnhancedEmbedding):
    def _process(self, span_pairs):
        return [el.length_difference() for el in span_pairs] 

class TypeEmbedding(EnhancedEmbedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.class2index = dict()

    def get_type(self, obj):
        if isinstance(obj, Span):
            return obj.label_
        if isinstance(obj, SpanGroup):
            return obj.attrs['type']
        else: 
            return obj.type


    def update_converter(self, objs):

            
        for el in objs:
            type_ = self.get_type(el)
            if type_ not in self.class2index.keys():
                self.class2index[type_] = len(self.class2index)        
    
    def _process(self, objs):
        self.update_converter(objs)
        return [self.class2index[self.get_type(el)] for el in objs] 

class LevenshteinEmbedding(EnhancedEmbedding):
    def _process(self, span_pairs):
        return [el.levenshtein_distance() for el in span_pairs]
    
class TokenDistanceEmbedding(EnhancedEmbedding):
    def _process(self, span_pairs):
        return [el.token_distance() for el in span_pairs]
        
class SentenceDistanceEmbedding(EnhancedEmbedding):
    def _process(self, span_pairs):
        return [el.sentence_distance() for el in span_pairs]

    


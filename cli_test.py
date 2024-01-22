
# #%%
# import torch
# from torch.nn import Module, Linear
# from torch.utils.data import Dataset, DataLoader

# # import lightning as pl
# from lightning import LightningModule, LightningDataModule
# from lightning.pytorch.cli import LightningCLI

# class Model(Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

# #%% 
# class MinimalLightningModule(LightningModule):
#     def __init__(self, dim):
#         super().__init__()
#         # self.model = Model(output_dim)
#         self.model = Linear(dim, dim)
        
#     # def _recursive_instantiate(self, config):
#     #     print('using _recursive_instantiate')
#     #     if isinstance(config, dict):
#     #         if 'function' in config:
#     #             config['function'] = eval(config['function'])    
#     #         if 'class' in config:
#     #             class_name = config['class']
#     #             class_config = config.get('config', {})
#     #             cls = eval(class_name)
            
#     #             return cls(**self.recursive_instantiate(class_config))
#     #         else:
#     #             return {k: self.recursive_instantiate(v) for k, v in config.items()}
#     #     return config
    
# #%%


# class SimpleDataset(Dataset):
#     def __init__(self, size=100, input_dim=10):

#         self.data = [torch.randn(input_dim) for _ in range(size)]
#         self.labels = [torch.randint(0, 2, (1,)) for _ in range(size)]


# # class SimpleDataModule(LightningDataModule):
# #     def __init__(self, batch_size = 32):
# #         super().__init__()

#     # def prepare_data(self):
#     #     # Data preparation logic here
#     #     pass

#     # def setup(self, stage=None):
#     #     # Dataset setup for each stage
#     #     if stage == 'fit':
#     #         self.train_dataset = SimpleDataset()
        
#     # def train_dataloader(self):
#     #     return DataLoader(self.train_dataset, batch_size=self.batch_size)


# # Use the data module
# # data_module = SimpleDataModule()

# #%% CLI
# class ELRECLI(LightningCLI):
    
#     def parse_arguments(self, parser, args):
#         print('using parse_arguments')
#         super().parse_arguments(parser, args)

#     def before_instantiate_classes(self):
#         print('using before_instantiate_classes')
    
#     def instantiate_classes(self):
#         print('using instantiate_classes')
#         super().instantiate_classes()
    
#     # def after_instantiate_classes(self):
        
#     #     print('using after_instantiate_classes')

#     #     model_params = self.config_init['model'] 
#     #     data_params = self.config_init['data']

#     #     # Update model config with data module properties
#     #     model_params['input_dims'] = data_params.get('entity_type_converter', None)
#     #     model_params['num_classes'] = data_params.get('num_classes', None)

#     #     # Re-instantiate the model with updated parameters
#     #     self.model = self.instantiate_class(self.config['model'], self.model_class, **model_params)

# imports
from torch.nn import Linear, Module
from torch.nn.functional import mish
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.cli import LightningCLI

# # lightning module
# class MinimalLightningModule(LightningModule):
#     def __init__(self, dim):
#         super().__init__()
#         self.model = Linear(dim, dim)


# def instantiate_class(class_name, **kwargs):
#     class_ = eval(class_name)
#     return class_(**kwargs)
def try_import(item):
    pass

def instantiate_class(class_info):
    if isinstance(class_info, dict) and "class" in class_info:
        # Extract class name and module name
        # class_name = class_info["class"]
        # module_name, class_name = class_name.rsplit(".", 1)
        # module = importlib.import_module(module_name)
        # class_ = getattr(module, class_name)

        class_ = eval(class_info['class'])

        # Check if there's a 'config' key and process it
        config = class_info.get("config", {})
        if isinstance(config, dict):
            for key, value in config.items():
                config[key] = instantiate_class(value)  # Recursive call

        # Instantiate the class with processed config
        return class_(**config)
    elif isinstance(class_info, dict) and "function" in class_info:
        return eval(class_info['function'])
    else:
        # Return as is if not a dictionary or no 'class' key
        return class_info

class Nested1(Module):
    def __init__(self, component1, component2):
        super().__init__()
        self.component1 = component1
        self.component2 = component2

class MLP1(Module):
    def __init__(self, output_dim, activation = mish):
        super().__init__()

        self.linear = LazyLinear(output_dim)
        self.activation = activation

        def forward(self, x):
            x = self.linear(x)
            x = self.activation(x)
            return x

# Lightning module
class MinimalLightningModule(LightningModule):
    def __init__(self, my_model):
        super().__init__()
        # print('type: ', type(my_model))
        # print(my_model)
        # model_class = my_model.get("class")
        # model_config = my_model.get("config", {})
        # self.model = instantiate_class(my_model)

class MinimalLightningModule2(LightningModule):
    def __init__(self, ner):
        super().__init__()


# CLI        
if __name__ == "__main__":
    # cli = LightningCLI(MinimalLightningModule, LightningDataModule)
    cli = LightningCLI(MinimalLightningModule2, LightningDataModule)
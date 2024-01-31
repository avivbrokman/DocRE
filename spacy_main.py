#%% libraries
from datetime import datetime
from importlib import import_module

from lightning.pytorch.cli import LightningCLI

from spacy_lightning_module import ELRELightningModule
from spacy_lightning_data_module import ELREDataModule


#%% CLI
class ELRECLI(LightningCLI):
    
    # def parse_arguments(self, parser, args):
    #     print('using parse_arguments')
    #     super().parse_arguments(parser, args)

    # def before_instantiate_classes(self):
    #     print('using before_instantiate_classes')
    
    # def instantiate_classes(self):
    #     print('using instantiate_classes')
    #     super().instantiate_classes()
    
    def _replace_string_with_def(self, class_info, key_name, module_names):
        for el in module_names:
            module = import_module(el)
            if hasattr(module, class_info[key_name]):
                class_info[key_name] = getattr(module, class_info[key_name])
                break

    def recursive_instantiate(self, class_info):
        print('using recursive instantiate')
        if isinstance(class_info, dict) and "class" in class_info:
            
            self._replace_string_with_def(class_info, 'class', ['spacy_modeling_classes', 'spacy_parameter_modules', 'torch.nn.functional'])

            # Check if there's a 'config' key and process it
            config = class_info.get("config", {})
            for key, value in config.items():
                config[key] = self.recursive_instantiate(value)

            # Instantiate the class with processed config
            return class_info['class'](**config)
        elif isinstance(class_info, dict) and "function" in class_info:
            # self._replace_string_with_def(class_info, 'function', ['modeling_classes', 'parameter_modules', 'torch.nn.functional'])
            # return class_info
            self._replace_string_with_def(class_info, 'function', ['spacy_modeling_classes', 'spacy_parameter_modules', 'torch.nn.functional'])
            
            return class_info['function']
        else:
            return class_info

    def after_instantiate_classes(self):
        
        print('using after_instantiate_classes')
        super().after_instantiate_classes()
        
        model = self.model

        if model.ner:
            model.ner = self.recursive_instantiate(model.ner_config)
        if model.clusterer:
            model.clusterer = self.recursive_instantiate(model.clusterer_config)
        if model.rc:
            model.rc = self.recursive_instantiate(model.rc_config)

        
        
        # model_params = self.config_init['model'] 
        # data_params = self.config_init['data']

        # # Update model config with data module properties
        # model_params['input_dims'] = data_params.get('entity_type_converter', None)
        # model_params['num_classes'] = data_params.get('num_classes', None)

        # # Re-instantiate the model with updated parameters
        # self.model = self.instantiate_class(self.config['model'], self.model_class, **model_params)

# #%% main
def main(config):
    cli = ELRECLI(ELRELightningModule, ELREDataModule, config = config)



#%% run
if __name__ == "__main__":
    # main(config)
    # cli = ELRECLI(ELRELightningModule, ELREDataModule)
    cli = LightningCLI(ELRELightningModule, ELREDataModule)


'''
python spacy_main.py fit --config ELRE_config.yaml

'''
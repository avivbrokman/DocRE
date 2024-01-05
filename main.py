#%% libraries
from datetime import datetime

from lightning.pytorch.cli import LightningCLI

from lightning_module import ELRELightningModule
from lightning_data_module import ELREDataModule

#%% imports for CLI
from parameter_modules import MLP2
from modeling_classes import NER, Coreference, RelationClassifier


#%% CLI
class ELRECLI(LightningCLI):
    def after_instantiate_classes(self):
        model_params = self.config_init['model']
        
        for el 
        
        model_params = self.config_init['model'] 
        data_params = self.config_init['data']

        # Update model config with data module properties
        model_params['input_dims'] = data_params.get('entity_type_converter', None)
        model_params['num_classes'] = data_params.get('num_classes', None)

        # Re-instantiate the model with updated parameters
        self.model = self.instantiate_class(self.config['model'], self.model_class, **model_params)

#%% main
def main(config):
    cli = LightningCLI(ELRELightningModule, ELREDataModule, config = config)


#%% run
if __name__ == "__main__":
    main()
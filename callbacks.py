#%% callback
import os
from lightning import Callback

class SaveAfterTrainCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        print('called on_train_epoch_end!')
        # Get the logger's save directory
        logger = trainer.logger
        dirpath = logger.save_dir

        # Get the version (experiment name or version number) from the logger
        version = trainer.logger.version if logger is not None else 'version_0'

        # Construct the checkpoint path
        filepath = f"{dirpath}/{version}/checkpoints"
        filename = f'checkpoint-epoch={trainer.current_epoch}.ckpt'
        checkpoint_path = os.path.join(filepath, filename)

        # Save the checkpoint
        trainer.save_checkpoint(checkpoint_path)

    # def on_validation_epoch_start(self, trainer, pl_module, outputs):
    #     print('called on_validation_epoch_start!')

    #     # Get the logger's save directory
    #     logger = trainer.logger
    #     dirpath = logger.save_dir

    #     # Get the version (experiment name or version number) from the logger
    #     version = trainer.logger.version if logger is not None else 'version_0'

    #     # Construct the checkpoint path
    #     filepath = f"{dirpath}/{version}/checkpoints"
    #     filename = f'checkpoint-epoch={trainer.current_epoch}.ckpt'
    #     checkpoint_path = os.path.join(filepath, filename)

    #     # Save the checkpoint
    #     trainer.save_checkpoint(checkpoint_path)

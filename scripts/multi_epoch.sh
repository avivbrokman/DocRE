#!/bin/bash

#navigate to root
# cd .

# Directory containing checkpoint files
#CHECKPOINT_DIR="lightning_logs/version_261/checkpoints"
#CHECKPOINT_DIR="lightning_logs/version_292/checkpoints"
# CHECKPOINT_DIR="lightning_logs/version_134/checkpoints"

# # Loop through each checkpoint file
# for CHECKPOINT_PATH in ${CHECKPOINT_DIR}/epoch=*.ckpt; do
#     # Extract epoch number
#     EPOCH=$(echo ${CHECKPOINT_PATH} | grep -oP 'epoch=\K\d+')

#     # Command to run for each checkpoint
#     COMMAND="python spacy_main.py validate --config ELRE_spacy_config.yaml --ckpt_path ${CHECKPOINT_PATH}"

#     # Use nohup to run the command in the background
#     nohup $COMMAND > "script_logs/validation_epoch_${EPOCH}.out" 2>&1 #& removed the & to prevent these scripts from moving the background, so they'll execute sequentially
    
# done



######
CHECKPOINT_DIR="lightning_logs/version_135/checkpoints"

# Loop through each checkpoint file
for CHECKPOINT_PATH in ${CHECKPOINT_DIR}/epoch=*.ckpt; do
    # Extract epoch number
    EPOCH=$(echo ${CHECKPOINT_PATH} | grep -oP 'epoch=\K\d+')

    # Command to run for each checkpoint
    COMMAND="python spacy_main.py validate --config ELRE_spacy_config.yaml --ckpt_path ${CHECKPOINT_PATH}"

    # Use nohup to run the command in the background
    nohup $COMMAND > "script_logs/validation_epoch_${EPOCH}.out" 2>&1 #& removed the & to prevent these scripts from moving the background, so they'll execute sequentially
    
done


# #######
# CHECKPOINT_DIR="lightning_logs/version_298/checkpoints"

# # Loop through each checkpoint file
# for CHECKPOINT_PATH in ${CHECKPOINT_DIR}/epoch=*.ckpt; do
#     # Extract epoch number
#     EPOCH=$(echo ${CHECKPOINT_PATH} | grep -oP 'epoch=\K\d+')

#     # Command to run for each checkpoint
#     COMMAND="python spacy_main.py validate --config ELRE_spacy_config.yaml --ckpt_path ${CHECKPOINT_PATH}"

#     # Use nohup to run the command in the background
#     nohup $COMMAND > "script_logs/validation_epoch_${EPOCH}.out" 2>&1 #& removed the & to prevent these scripts from moving the background, so they'll execute sequentially
    
# done
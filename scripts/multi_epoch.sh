#!/bin/bash


function zip_performance() {
    local RUN_NAME="$1"
    if [ $# -eq 2 ]; then
        local COREF_CUTOFF=$2
    fi

    local RUN_DIR="lightning_logs/${RUN_NAME}"
    local CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
    local CONFIG_PATH="${RUN_DIR}/config.yaml"

    local VALIDATION_DIR="${RUN_DIR}/validation"

    if [ $# -eq 2 ]; then
        VALIDATION_DIR="${VALIDATION_DIR}_coref_cutoff_${COREF_CUTOFF}"
    fi

    echo "starting zip stuff"
    # Temporary directory for flattening the file structure
    
    TEMP_DIR="temp_json_files"
    mkdir -p "$TEMP_DIR"

    echo "made temp dir"

    # Find and copy .json files to the temporary directory
    find "$VALIDATION_DIR" -name "*.json" -exec cp {} "$TEMP_DIR/" \;

    echo "found"

    # Create an archive from the temporary directory contents
    
    if [ $# -eq 1 ]; then
        tar -czvf "${RUN_DIR}/${RUN_NAME}.tar.gz" -C "$TEMP_DIR" .
    else
        tar -czvf "${RUN_DIR}/${RUN_NAME}_coref_cutoff_${COREF_CUTOFF}.tar.gz" -C "$TEMP_DIR" .
    fi

    echo "zipped"

    # Clean up: Remove the temporary directory
    rm -rf "$TEMP_DIR"

    echo "finished"
}




function validate_checkpoints() {
    local RUN_NAME="$1"

    if [ $# -eq 2 ]; then
        local COREF_CUTOFF=$2
    fi

    local RUN_DIR="lightning_logs/${RUN_NAME}"
    local CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
    local CONFIG_PATH="${RUN_DIR}/config.yaml"

    local VALIDATION_DIR="${RUN_DIR}/validation"

    if [ $# -eq 2 ]; then
        VALIDATION_DIR="${VALIDATION_DIR}_coref_cutoff_${COREF_CUTOFF}"
        cp $CONFIG_PATH temp_config.yaml
        CONFIG_PATH="temp_config.yaml"
        # yq e ".model.clusterer_config.config.coref_cutoff = ${COREF_CUTOFF}" -i temp_config.yaml
        yq write -i temp_config.yaml "model.clusterer_config.config.coref_cutoff" "${COREF_CUTOFF}"
    fi

    mkdir -p "$VALIDATION_DIR"

    # Loop through each checkpoint file
    for CHECKPOINT_PATH in ${CHECKPOINT_DIR}/epoch=*.ckpt; do
        # Extract epoch number
        local EPOCH=$(echo ${CHECKPOINT_PATH} | grep -oP 'epoch=\K\d+')
        echo "epoch ${EPOCH}"
        # Make SAVE_DIR
        # local SAVE_DIR="${VALIDATION_DIR}/epoch_${EPOCH}"
        # mkdir $SAVE_DIR
        local VERSION="epoch_${EPOCH}"

        # Command to run for each checkpoint
        # local COMMAND="python spacy_main.py validate --config ${CONFIG_PATH} --ckpt_path ${CHECKPOINT_PATH} --trainer.logger.init_args.save_dir $(pwd) --trainer.logger.init_args.name ${VALIDATION_DIR} --trainer.logger.init_args.version ${VERSION}"

        # if [ $# -eq 2 ]; then
        #     COMMAND="$COMMAND --model.clusterer_config.config.coref_cutoff ${COREF_CUTOFF}"
        # fi

        local COMMAND="python spacy_main.py validate --config $CONFIG_PATH --ckpt_path ${CHECKPOINT_PATH} --trainer.logger.init_args.save_dir $(pwd) --trainer.logger.init_args.name ${VALIDATION_DIR} --trainer.logger.init_args.version ${VERSION}"

        echo $COMMAND

        # Use nohup to run the command in the background
        nohup $COMMAND > "script_logs/validation_epoch_${EPOCH}.out" 2>&1

        echo "finished epoch"
    done

    if [ $# -eq 2 ]; then
        rm temp_config.yaml
    fi

    if [ $# -eq 1 ]; then
        zip_performance $RUN_NAME
    else
        zip_performance $RUN_NAME $COREF_CUTOFF
    fi
}


# Call the function with different RUN_NAME values
# validate_checkpoints "BioRED_ner_p0.01"
# validate_checkpoints "BioRED_ner_p0.2"
# validate_checkpoints "BioRED_ner_p0.5"

# validate_checkpoints "BioRED_cluster_p0.2" 0.3
# validate_checkpoints "BioRED_cluster_p0.2" 0.4
# validate_checkpoints "BioRED_cluster_p0.2" 0.6
# validate_checkpoints "BioRED_cluster_p0.2" 0.7


# zip_performance "BioRED_cluster_p0.2"
# vzip_performance "BioRED_cluster_p0.2" 0.3
# zip_performance "BioRED_cluster_p0.2" 0.4
# zip_performance "BioRED_cluster_p0.2" 0.6
# zip_performance "BioRED_cluster_p0.2" 0.7

# validate_checkpoints "BioRED_e2e_p0.01"
# validate_checkpoints "BioRED_ner_mlp2"
# validate_checkpoints "BioRED_cluster_mlp2"
# validate_checkpoints "BioRED_rc_mlp2"
validate_checkpoints "BioRED_e2e_mlp2"
# validate_checkpoints "BioRED_ner_mlp2"
# validate_checkpoints "BioRED_e2e_p0.5"
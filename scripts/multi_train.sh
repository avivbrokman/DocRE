#!/bin/bash


function train(){
    local TASK="$1"
    local DATASET="$2"

    local RUN_NAME="${DATASET}_${TASK}"

    if [ $# -eq 3 ]; then
        local SUFFIX="$3"
        RUN_NAME="${RUN_NAME}_${SUFFIX}"
    fi

    python spacy_main.py fit --config ELRE_spacy_config.yaml --model.task $TASK --model.dataset_name $DATASET --data.dataset_name $DATASET --trainer.logger.init_args.save_dir $(pwd) --trainer.logger.init_args.name lightning_logs --trainer.logger.init_args.version $RUN_NAME 
}


# train ner BioRED
# train cluster BioRED
# train rc BioRED
train e2e BioRED v2

# train ner CDR
# train cluster CDR
# train rc CDR
# train e2e CDR
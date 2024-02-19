#!/bin/bash


function train(){
    local TASK="$1"
    local DATASET="$2"

    local RUN_NAME="${DATASET}_${TASK}"

    if [ $# -eq 3 ]; then
        local SUFFIX="$3"
        RUN_NAME="${RUN_NAME}_${SUFFIX}"
    fi


    # local CONFIG="ELRE_spacy_config.yaml"
    local CONFIG="mlp2_config.yaml"

    python spacy_main.py fit --config $CONFIG --model.task $TASK --model.dataset_name $DATASET --data.dataset_name $DATASET --trainer.logger.init_args.save_dir $(pwd) --trainer.logger.init_args.name lightning_logs --trainer.logger.init_args.version $RUN_NAME 
}


# train ner BioRED hash_fix
train cluster BioRED hash_fix_2   
# train rc BioRED mlp2_p0.6
# train rc BioRED mlp2_p0.6
# train rc BioRED mlp2_p0.7
# train e2e BioRED hash_fix

# train ner CDR hash_fix
# train cluster CDR hash_fix
# train rc CDR hash_fix
# train e2e CDR hash_fix
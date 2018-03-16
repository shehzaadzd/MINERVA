#!/usr/bin/env bash

source $1
export PYTHONPATH="."
gpu_id=0
cmd="python code/model/trainer.py --base_output_dir $base_output_dir --path_length $path_length --hidden_size $hidden_size --embedding_size $embedding_size \
    --batch_size $batch_size --beta $beta --Lambda $Lambda --use_entity_embeddings $use_entity_embeddings \
    --train_entity_embeddings $train_entity_embeddings --train_relation_embeddings $train_relation_embeddings \
    --data_input_dir $data_input_dir --vocab_dir $vocab_dir --model_load_dir $model_load_dir --load_model $load_model --total_iterations $total_iterations --nell_evaluation $nell_evaluation"



echo "Executing $cmd"

CUDA_VISIBLE_DEVICES=$gpu_id $cmd

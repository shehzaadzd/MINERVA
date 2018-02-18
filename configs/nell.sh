#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/nell-995/"
vocab_dir="datasets/data_preprocessed/nell-995/vocab"
total_iterations=3000
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.05
Lambda=0.02
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/nell/worksfor"
load_model=1
model_load_dir="output/nell/25dd_3_0.05_50_0.02/model/model.ckpt"
nell_evaluation=0

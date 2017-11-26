#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/grid/grid_10_8/"
vocab_dir="datasets/data_preprocessed/grid/grid_10_8/vocab"
total_iterations=1000
path_length=10
hidden_size=25
embedding_size=25
batch_size=4096
beta=0.02
Lambda=0.0
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/grid_10/"
load_model=0
model_load_dir="null"

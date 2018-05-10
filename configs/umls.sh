#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/umls/"
vocab_dir="datasets/data_preprocessed/umls/vocab"
total_iterations=2000
path_length=2
hidden_size=50
embedding_size=50
batch_size=256
beta=0.05
Lambda=0.05
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/umls/"
load_model=1
model_load_dir="/home/sdhuliawala/logs/RL-Path-RNN/uuuu/8fe2_2_0.06_10_0.02/model/model.ckpt"
nell_evaluation=0

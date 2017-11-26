#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/countries_S2/"
vocab_dir="datasets/data_preprocessed/countries_S2/vocab"
total_iterations=1000
path_length=2
hidden_size=15
embedding_size=15
batch_size=256
beta=0.02
Lambda=0.0
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/countries_s2"
load_model=0
#load_model=1
#model_load_dir="saved_models/countries_s2/model.ckpt"

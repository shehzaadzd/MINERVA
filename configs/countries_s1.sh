#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/countries_S1/"
vocab_dir="datasets/data_preprocessed/countries_S1/vocab"
total_iterations=1000
path_length=2
hidden_size=25
embedding_size=25
batch_size=256
beta=0.05
Lambda=0.05
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/countries_s1/"
load_model=0
model_load_dir="null"
nell_evaluation=0
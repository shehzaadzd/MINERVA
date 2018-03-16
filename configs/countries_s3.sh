#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/countries_S3/"
vocab_dir="datasets/data_preprocessed/countries_S3/vocab"
total_iterations=1000
path_length=3
hidden_size=2
embedding_size=2
batch_size=128
beta=0.1
Lambda=0.02
use_entity_embeddings=1
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/countries_s3/"
model_load_dir="nothing"
load_model=0
nell_evaluation=0

#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/countries_S3/"
vocab_dir="datasets/data_preprocessed/countries_S3/vocab"
total_iterations=1000
path_length=3
hidden_size=15
embedding_size=15
batch_size=150
beta=0.05
Lambda=0.0
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/countries_s3/"
model_load_dir="nothing"
load_model=0


import json
import csv
import os
import  numpy as np
task_dir = '/Users/shehzaad/projects/DeepPath/NELL-995/tasks/concept_teamplaysinleague'

task_name = 'teamplaysinleague'

entity_vocab={}
relation_vocab = {}
entity_vocab['PAD'] = len(entity_vocab)
entity_vocab['UNK'] = len(entity_vocab)
relation_vocab['PAD'] = len(relation_vocab)
relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
relation_vocab['NO_OP'] = len(relation_vocab)
relation_vocab['UNK'] = len(relation_vocab)

root_dir = "../../../"
output_dir = root_dir+'/'+"datasets/data_preprocessed/"+task_name
vocab_dir = output_dir +'/vocab'
os.makedirs(vocab_dir)
t_name  = ''


with open(output_dir+'/train.txt', 'w') as out_file:
    with open(output_dir + '/dev.txt', 'w') as dev:
        with open(task_dir+'/train_pos') as raw_file:
            csv_file = csv.reader(raw_file, delimiter='\t')
            relation_counter = len(relation_vocab)
            entity_counter = len(entity_vocab)
            for line in csv_file:
                e1, e2, r= line
                t_name = r
                if e1 not in entity_vocab:
                    entity_vocab[e1] = entity_counter
                    entity_counter += 1
                if e2 not in entity_vocab:
                    entity_vocab[e2] = entity_counter
                    entity_counter += 1
                if r not in relation_vocab:
                    relation_vocab[r] = relation_counter
                    relation_counter += 1

                out_file.write(e1+'\t'+r+'\t'+e2+'\n')
                if np.random.normal() > 0.2:
                    dev.write(e1+'\t'+r+'\t'+e2+'\n')

with open(output_dir + '/graph.txt', 'w') as out_file:
    with open(task_dir+'/graph.txt') as raw_file:
        csv_file = csv.reader(raw_file, delimiter='\t')
        relation_counter = len(relation_vocab)
        entity_counter = len(entity_vocab)
        for line in csv_file:
            e1, r, e2 = line
            if e1 not in entity_vocab:
                entity_vocab[e1] = entity_counter
                entity_counter += 1
            if e2 not in entity_vocab:
                entity_vocab[e2] = entity_counter
                entity_counter += 1
            if r not in relation_vocab:
                relation_vocab[r] = relation_counter
                relation_counter += 1
            out_file.write(e1+'\t'+r+'\t'+e2+'\n')
    with open(task_dir + '/train_pos') as raw_file:
        csv_file = csv.reader(raw_file, delimiter='\t')
        relation_counter = len(relation_vocab)
        entity_counter = len(entity_vocab)
        for line in csv_file:
            e1, e2, r = line
            t_name = r
            if e1 not in entity_vocab:
                entity_vocab[e1] = entity_counter
                entity_counter += 1
            if e2 not in entity_vocab:
                entity_vocab[e2] = entity_counter
                entity_counter += 1
            if r not in relation_vocab:
                relation_vocab[r] = relation_counter
                relation_counter += 1

            out_file.write(e1 + '\t' + r + '\t' + e2 + '\n')

with open(output_dir+'/test.txt', 'w') as out_file:
    with open(task_dir+'/test.pairs') as raw_file:
        relation_counter = len(relation_vocab)
        entity_counter = len(entity_vocab)
        for line in raw_file:

            if line.strip()[-1] == '+':
                print "here"
                ee = line.strip()[:-3]
                e1, e2 = ee.split(',')
                # import pdb
                # pdb.set_trace()
                e1 = e1[6:]
                e2 = e2[6:]
                if e1 not in entity_vocab:
                    entity_vocab[e1] = entity_counter
                    entity_counter += 1
                if e2 not in entity_vocab:
                    entity_vocab[e2] = entity_counter
                    entity_counter += 1
                if task_name not in relation_vocab:
                    relation_vocab[task_name] = relation_counter
                    relation_counter += 1

                out_file.write(e1+'\t'+t_name+'\t'+e2+'\n')


with open(vocab_dir + '/entity_vocab.json', 'w') as fout:
    json.dump(entity_vocab, fout)

with open(vocab_dir + '/relation_vocab.json', 'w') as fout:
    json.dump(relation_vocab, fout)




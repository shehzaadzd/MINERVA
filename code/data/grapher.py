from collections import defaultdict
import logging
import numpy as np
import csv

logger = logging.getLogger(__name__)


class RelationEntityGrapher:
    def __init__(self, triple_store, relation_vocab, entity_vocab, max_num_actions):

        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.triple_store = triple_store
        self.relation_vocab = relation_vocab
        self.entity_vocab = entity_vocab
        self.store = defaultdict(list)
        self.array_store = np.ones((len(entity_vocab), max_num_actions, 2), dtype=np.dtype('int32'))
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD
        self.masked_array_store = None

        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])
        self.create_graph()
        print("KG constructed")

    def create_graph(self):
        with open(self.triple_store) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                e1 = self.entity_vocab[line[0]]
                r = self.relation_vocab[line[1]]
                e2 = self.entity_vocab[line[2]]
                self.store[e1].append((r, e2))

        for e1 in self.store:
            num_actions = 1
            self.array_store[e1, 0, 1] = self.relation_vocab['NO_OP']
            self.array_store[e1, 0, 0] = e1
            for r, e2 in self.store[e1]:
                if num_actions == self.array_store.shape[1]:
                    break
                self.array_store[e1,num_actions,0] = e2
                self.array_store[e1,num_actions,1] = r
                num_actions += 1
        del self.store
        self.store = None

    def return_next_actions(self, current_entities, start_entities, query_relations, answers, all_correct_answers, last_step, rollouts):
        ret = self.array_store[current_entities, :, :].copy()
        for i in range(current_entities.shape[0]):
            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                mask = np.logical_and(relations == query_relations[i] , entities == answers[i])
                ret[i, :, 0][mask] = self.ePAD
                ret[i, :, 1][mask] = self.rPAD
            if last_step:
                entities = ret[i, :, 0]
                relations = ret[i, :, 1]

                correct_e2 = answers[i]
                for j in range(entities.shape[0]):
                    if entities[j] in all_correct_answers[i/rollouts] and entities[j] != correct_e2:
                        entities[j] = self.ePAD
                        relations[j] = self.rPAD

        return ret

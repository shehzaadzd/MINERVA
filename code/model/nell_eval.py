from __future__ import division
import csv
from collections import defaultdict
import random
import numpy as np



def nell_eval(model_answers, correct_answers):
    answers = defaultdict(dict)
    possible_answers_correct = defaultdict(set)
    possible_answers_incorrect = defaultdict(set)
    with open(model_answers, 'r') as answers_file:
        triples = csv.reader(answers_file, delimiter='\t')
        for line in triples:
            e1,e2, p = line
            p = float(p)

            if e2 in answers[e1]:
                if answers[e1][e2] < p:
                    answers[e1][e2] = p
            else:
                answers[e1][e2] = p

    with open(correct_answers) as answer_file:
        for line in answer_file:
            l = line.strip()
            label = 1 if l[-1] == '+' else 0
            e1,e2 = l.split(':')[0].split(',')
            e1 = e1[6:]
            e2 = e2[6:]
            if label == 0:
                possible_answers_incorrect[e1].add(e2) #all incorrect answers
            if label == 1:
                possible_answers_correct[e1].add(e2) # all incorrect answers

    #evaluate
    means = []
    for e1 in answers:

        ranks = []
        count_correct = 0
        count_incorrect = 0
        all_answers = sorted(answers[e1], key=answers[e1].get, reverse=True)

        total_answers = len(possible_answers_incorrect[e1]) + len(possible_answers_correct[e1])
        for a in all_answers:
            if a in possible_answers_correct[e1]:
                count_correct += 1
                ranks.append(count_correct+count_incorrect)
                if count_correct == len(possible_answers_correct):
                    break
            if a in possible_answers_incorrect[e1]:
                count_incorrect +=1

        assert len(possible_answers_correct[e1]) > 0
        for i in range(len(possible_answers_correct[e1]) - count_correct):
            ranks.append(random.randint(count_correct+count_incorrect+1+i, total_answers))

        mean = 0
        for i,r in enumerate(ranks):
            mean += (i+1)/r
        mean /= len(ranks)
        means.append(mean)

    print sum(means)/len(means)


def nell_eval_victoria(model_answers, correct_answers):
    test_data_path = correct_answers
    test_prediction_path = model_answers
    f = open(test_data_path)
    test_data = f.readlines()
    f.close()

    # load prediction scores
    preds = {}
    with open(test_prediction_path) as f:
        for line in f:
            e1,e2, score = line.strip().split()
            score = float(score)
            if (e1, e2) not in preds:
                preds[(e1, e2)] = score
            else:
                if preds[(e1,e2)] < score:
                    preds[(e1,e2)] = score

    def get_pred_score(e1, e2):
        if (e1, e2) in preds:
            return preds[(e1,e2)]
        else:
            return -np.inf
    test_pairs = defaultdict(lambda : defaultdict(int))
    for line in test_data:
        e1 = line.split(',')[0].replace('thing$','')
        e2 = line.split(',')[1].split(':')[0].replace('thing$','')

        label = 1 if line[-2] == '+' else 0
        test_pairs[e1][e2] = label
    aps = []


    score_all = []

    # calculate MAP
    for e1 in test_pairs:
        y_true = []
        y_score = []
        for  e2 in test_pairs[e1]:
            score = get_pred_score(e1, e2)
            score_all.append(score)
            y_score.append(score)
            y_true.append(test_pairs[e1][e2])
        count = list(zip(y_score, y_true))
        count.sort(key=lambda x: x[0], reverse=True)

        ranks = []
        correct = 0
        for idx_, item in enumerate(count):
            if item[1] == 1:
                correct += 1
                ranks.append(correct / (1.0 + idx_))
        if len(ranks) == 0:
            ranks.append(0)
        aps.append(np.mean(ranks))
    mean_ap = np.mean(aps)
    print('MINERVA MAP: {} ({} queries evaluated)'.format( mean_ap, len(aps)))

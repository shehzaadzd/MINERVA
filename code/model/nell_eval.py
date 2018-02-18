from __future__ import division
import csv
from collections import defaultdict
import random
import  sys



def nell_eval(model_answers, correct_answers):


    answers = defaultdict(dict)
    possible_answers_correct = defaultdict(set)
    possible_answers_incorrect = defaultdict(set)
    with open(model_answers, 'r') as answers_file:
        triples = csv.reader(answers_file, delimiter='\t')
        for line in triples:
            e1,e2, p = line #e1, e2, score
            p = float(p)
            # if e1 == 'concept_athlete_yunel_escobar':
            #     import pdb
            #
            #     pdb.set_trace()
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

        candidate_answers = []
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

        # print len(possible_answers_correct[e1])
        mean = 0
        for i,r in enumerate(ranks):
            mean += (i+1)/r
        mean /= len(ranks)
        means.append(mean)

    print sum(means)/len(means)


def nell_eval_all(model_answers, correct_answers):
    all_model_answers = []
    all_correct_answers = []

    answers = defaultdict(dict)
    possible_answers_correct = defaultdict(set)
    possible_answers_incorrect = defaultdict(set)
    with open(model_answers, 'r') as answers_file:
        triples = csv.reader(answers_file, delimiter='\t')
        for line in triples:
            e1, e2, p = line  # e1, e2, score
            p = float(p)
            # if e1 == 'concept_athlete_yunel_escobar':
            #     import pdb
            #
            #     pdb.set_trace()
            if e2 in answers[e1]:
                if answers[e1][e2] < p:
                    answers[e1][e2] = p
            else:
                answers[e1][e2] = p

    with open(correct_answers) as answer_file:
        for line in answer_file:
            l = line.strip()
            label = 1 if l[-1] == '+' else 0
            e1, e2 = l.split(':')[0].split(',')
            e1 = e1[6:]
            e2 = e2[6:]
            if label == 0:
                possible_answers_incorrect[e1].add(e2)  # all incorrect answers
            if label == 1:
                possible_answers_correct[e1].add(e2)  # all incorrect answers

    # evaluate
    means = []
    for e1 in answers:

        candidate_answers = []
        ranks = []
        count_correct = 0
        count_incorrect = 0
        all_answers = sorted(answers[e1], key=answers[e1].get, reverse=True)

        total_answers = len(possible_answers_incorrect[e1]) + len(possible_answers_correct[e1])
        for a in all_answers:
            if a in possible_answers_correct[e1]:
                count_correct += 1
                ranks.append(count_correct + count_incorrect)
                if count_correct == len(possible_answers_correct):
                    break
            if a in possible_answers_incorrect[e1]:
                count_incorrect += 1

        assert len(possible_answers_correct[e1]) > 0
        for i in range(len(possible_answers_correct[e1]) - count_correct):
            ranks.append(random.randint(count_correct + count_incorrect + 1 + i, total_answers))

        # print len(possible_answers_correct[e1])
        mean = 0
        for i, r in enumerate(ranks):
            mean += (i + 1) / r
        mean /= len(ranks)
        means.append(mean)

    print "MAP:"
    print sum(means) / len(means)


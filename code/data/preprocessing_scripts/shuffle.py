nbr = {}
new_train = set()
with open('../../../datasets/data_preprocessed/countries_S2/graph.txt') as graph:
    for line in graph:
        e1, r, e2 = line.strip().split('\t')
        if r=='neighborOf':
            nbr[e1] = True
        if e2 in ["northern_america",
"eastern_europe",
"australia_and_new_zealand",
"melanesia",
"micronesia",
"eastern_africa",
"southern_asia",
"eastern_asia",
"south_america",
"central_europe",
"western_asia",
"northern_africa",
"western_africa",
"northern_europe",
"middle_africa",
"caribbean",
"polynesia",
"western_europe",
"southern_europe",
"central_america",
"southern_africa",
"central_asia",
"south-eastern_asia"
]:
            new_train.add((e1,r,e2))
        # new_train.add((e2, "_"+r, e1))
# with open('../../../datasets/data_preprocessed/umls_inv/dev.txt') as graph:
#     for line in graph:
#         e1, r, e2 = line.strip().split('\t')
#         new_train.add((e1,r,e2))
with open('../../../datasets/data_preprocessed/countries_S2/train.txt', 'w') as graph:
    for line in new_train:
        e1, r, e2 = line
        if e1 in nbr:
            graph.write(e1+"\t"+r+"\t"+e2+"\n")

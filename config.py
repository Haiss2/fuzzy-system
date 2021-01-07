# data
train_path = 'data/glass_training_data.csv'
test_path = 'data/glass_test_data.csv'
# full_path = 'data/glass.csv'
full_path = 'data/ecoli.csv'

# export file
clusters_path = 'export/Clusters.csv'
rule_path = 'export/Rules.csv'
fmc_path = 'export/Fmc.csv'
editFmc_path = 'export/Edit.csv'

num_classes = 4

# clustering
k_mean = 4
m_fuzzy = 2
e_fcm = 0.001

# make rule
e_rule = 0.007
num_rules = 80

# fixed 4 gia tu: V, M, P, L
ha_tree_deep = 3

# cross validation iterators
k_fold = 3
shuffle = False
   
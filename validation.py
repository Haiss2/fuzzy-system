import time
import config as cf
from code import csv_processor as csv
from code.clustering import Clustering
from code.make_rule import Rule
from code.predict import Predict

from sklearn.model_selection import KFold
import numpy as np

def predict(data, slug):
    now = time.time()
    x = Predict(data, clusters, rules)

    corrects = 0
    edit = []
    editCollection = []

    for ix, record in enumerate(data):
        if int(record[0]) == x.predict(record, rules)["predict"]:
            corrects += 1
        else:
            pr_rules = x.predict(record, rules)["rule"]
            cr_rules = x.get_rule_truth(record)["rule"]
            edit += x.detect_cluster(pr_rules, cr_rules, record[1: len(record) ])
    print('\n  __{} >> Predict time: {:.2f}s'.format(slug, time.time() - now))
    print("  __Correct: {}, Total: {}, Accuracy: {:.2f}%\n".format(corrects, len(data), 100*corrects / len(data)))

def train_func(data):
    now = time.time()

    # clustering and save
    clusters  = Clustering(data).clusters
    csv.write_file(cf.clusters_path, clusters)

    # make_rule and save
    rules = Rule(data, clusters).colection_rules()
    csv.write_file(cf.rule_path, rules)

    # Predict and save data
    x = Predict(data, clusters, rules)

    corrects = 0
    edit = []
    editCollection = []

    for ix, record in enumerate(data):
        if int(record[0]) == x.predict(record, rules)["predict"]:
            corrects += 1
        else:
            pr_rules = x.predict(record, rules)["rule"]
            cr_rules = x.get_rule_truth(record)["rule"]
            edit += x.detect_cluster(pr_rules, cr_rules, record[1: len(record) ])
    print('\n  __Predict time: {:.2f}s'.format(time.time() - now))
    print("\n  __Correct: {}, Total: {}, Accuracy: {:.2f}%\n".format(corrects, len(data), 100*corrects / len(data)))

if __name__ == "__main__":

    # read data
    data = csv.read_file(cf.full_path, 'float')

    kf = KFold(n_splits=cf.k_fold, shuffle=False)
    
    result_data = [1 for i in range(cf.num_classes) ]
    for i in range(cf.num_classes):
        data_class = []
        for j in data:
            if j[0] == i+1: data_class.append(j)
        result_data[i] = (data_class, list(kf.split(data_class)))
    
    for i in range(cf.k_fold):
        train = []
        test = []
        for j in range(cf.num_classes):
            data_class = result_data[j][0]
            # result_data[0][1][2][3]
            # 1: class, 2: 1 is kf.split object, 2: fold, 3: train or test
            for train_ids in result_data[j][1][i][0]:
                train.append(data_class[train_ids])
            for test_ids in result_data[j][1][i][1]:
                test.append(data_class[test_ids])
        
        train_func(train)
        # read data
        predict(test, "Test Data")


    
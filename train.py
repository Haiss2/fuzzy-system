import time
import config as cf
from code import csv_processor as csv
from code.clustering import Clustering
from code.make_rule import Rule
from code.predict import Predict

if __name__ == "__main__":
    _1st = time.time()

    # read data
    data = csv.read_file(cf.train_path, 'float')

    # clustering and save
    clusters  = Clustering(data).clusters
    csv.write_file(cf.clusters_path, clusters)
    _2nd = time.time()
    print('  __Clustering time: {:.2f}s'.format(_2nd - _1st))

    # make_rule and save
    rules = Rule(data, clusters).colection_rules()
    csv.write_file(cf.rule_path, rules)
    _3rd = time.time()
    print('  __Making rule time: {:.2f}s'.format(_3rd - _2nd))

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

    _4th = time.time()
    print('  __Predict time: {:.2f}s'.format(_4th - _3rd))
    
    for e in edit:
        if 0.5 < e[2] and e[2] < 0.99 and e[2] + e[4] == 1:
            editCollection.append(e)

    def editCollectionSort(x):
        return x[2]

    editCollection.sort(key=editCollectionSort)
    csv.write_file(cf.editFmc_path, editCollection)

    # default fcm_path
    fmc = [[0.5 for i in range(2*cf.k_mean)] for i in range(len(data[0]) - 1)]
    csv.write_file(cf.fmc_path, fmc)

    print('  __T1FS Training time: {:.2f}s'.format(time.time() - _1st))
    print("\n  __Correct: {}, Total: {}, Accuracy: {:.2f}%\n".format(corrects, len(data), 100*corrects / len(data)))
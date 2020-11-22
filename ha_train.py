from ha_predict import Predict
import config as cf
import code.csv_processor as csv
from code.ha_tree import HaTree
import time

if __name__ == "__main__":
    data = csv.read_file(cf.train_path, 'float')
    clusters = csv.read_file(cf.clusters_path, 'float')
    rules = csv.read_file(cf.rule_path, 'float')
    edit = csv.read_file(cf.editFmc_path, 'float')

    _start = time.time()
    
    fmc = csv.read_file(cf.fmc_path, 'float')
    
    print("\n  -----> starting")
    x = Predict(data, clusters, rules, fmc)
    correct = x.num_corrects()
    print("\n")

    for attr, flase, u_fasle, true, u_true in edit:
        
        fmc = csv.read_file(cf.fmc_path, 'float')        
        if true > flase:
            fmc[attr][2*true] = u_true/u_fasle*fmc[attr][2*true - 1]
        else:
            fmc[attr][2*true + 1] = u_true/u_fasle*fmc[attr][2*true + 2]
        
        x = Predict(data, clusters, rules, fmc)
        y = x.num_corrects()
        if correct <= y:
            print("  -----> edited", (attr, flase, u_fasle, true, u_true ), "\n")
            correct = y
            csv.write_file(cf.fmc_path, fmc)
        else: 
            print("  -----> rejected", (attr, flase, u_fasle, true, u_true ), "\n")
    print("  Training Time: {:.2f}s, Best Accuracy: {:.3f}%\n".format(time.time() - _start, correct*100/len(data)))
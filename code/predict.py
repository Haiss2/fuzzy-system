import config as cf
import code.csv_processor as csv

class Predict:
    def __init__(self, data, cluster, rule):
        self.data = data
        self.cluster = cluster
        self.rules = rule
        self.num_properties = len(self.data[0]) - 1

    def do_thuoc(self, x, c1, c2, c3):
        if x >= c3 or x <= c1: 
            return 0
        if c1 < x and x <= c2:
            return (x - c1) / (c2 - c1)
        if c2 < x and x < c3:
            return (c3 - x) / (c3 - c2)

    # x = data, s = index of Cluseter, C = cluster of attribute
    def do_thuoc_set(self, x, s, C):
        if s == 0:
            return self.do_thuoc(x, 0, C[0], C[1])
        if s == cf.k_mean - 1:
            return self.do_thuoc(x, C[s - 1], C[s], 100)
        return self.do_thuoc(x, C[s-1], C[s], C[s+1])

    def predict(self, t, r): # t la ban ghi
        rule_burn = [min( [self.do_thuoc_set(t[i+1], rule[i+1], self.cluster[i]) for i in range(self.num_properties)]) for rule in r]
        do_thuoc_sum = [sum( [self.do_thuoc_set(t[i+1], rule[i+1], self.cluster[i])**0.5 for i in range(self.num_properties)]) for rule in r]
        maxx = max(rule_burn)
        summ = 0
        ids = []
        for (i, j) in enumerate(rule_burn):
            if j == maxx:
                ids.append(i)
        if len(ids) == 1:
            id = rule_burn.index(maxx)
        else:
            summ = [do_thuoc_sum[i] for i in ids]
            id = do_thuoc_sum.index(max(summ))
        return {
            "predict": int(r[id][0]),
            "rule_id": id,
            "rule": r[id][1:self.num_properties + 1],
            "sum": len(ids)
        }

    def get_rule_truth(self, t): # t la ban ghi
        sub_rules = []
        get_first = False
        first_id = 0
        for i, j in enumerate(self.rules):
            if int(j[0]) == int(t[0]):
                sub_rules.append(j)
                if not get_first:
                    first_id = i
                    get_first = True
        correct_rule = self.predict(t, sub_rules)
        return {
            "predict": t[0],
            "rule_id": correct_rule["rule_id"] + first_id,
            "rule": correct_rule["rule"]
        }
    
    
    def detect_cluster(self, a, b, data):
        result = []
        for i in range(self.num_properties):
            if a[i] != b[i] and abs(a[i] - b[i]) == 1:
                result += [[i, a[i], self.do_thuoc_set(data[i], a[i], self.cluster[i]), b[i], self.do_thuoc_set(data[i], b[i], self.cluster[i])]]
        return result

if __name__ == "__main__":
        
    # read data
    data = csv.read_file(cf.train_path, 'float')
    clusters = csv.read_file(cf.clusters_path, 'float')
    rules = csv.read_file(cf.rule_path, 'float')

    # clustering and save
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

    print("Corrects:", corrects)
    print("Total:", len(data))
    print("Accuracy {:.2f}%".format(corrects/len(data) * 100))
    print("Watch edit rule in", cf.editFmc_path)

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
from read_data import read_file
import config as cf
import time
cluster_path = 'Cluster.csv'
rule_path = 'Rule.csv'
fmc_path = 'Fmc.csv'
Fmc = read_file(fmc_path, 'float')
Fmc2 = [[(i[0], i[1]), (i[2], i[3]),(i[4], i[5]),(i[6], i[7]),(i[8], i[9])] for i in Fmc]
# print(Fmc2)
# exit()
Cluster_center = read_file(cluster_path, 'float')
Rule_set = read_file(rule_path, 'float')
path = 'TrainingData.csv'
data = read_file(path, 'float')
data_size = len(data)
num_properties = len(data[0]) - 1
k_mean = cf.k_mean
m_fuzzy = cf.m_fuzzy
e_fcm = cf.e_fcm
e_rule = cf.e_rule


def do_thuoc(x, c1, c2, c3, fmc):
    if x >= c3 or x <= c1: 
        return 0
    if c1 < x and x <= c2:
        if c1 == 0 and cf.special:
            return 1
        return min(fmc[0] * (x - c1) / (c2 - c1), 1)
    if c2 < x and x < c3:
        if c3 == 100 and cf.special:
            return 1
        return min(fmc[1] * (c3 - x) / (c3 - c2), 1)

# predict
def do_thuoc_set(x, s, C, F=[(1,1),(1,1),(1,1),(1,1),(1,1)]):
    if s == 0:
        return do_thuoc(x, 0, C[0], C[1], F[s])
    if s == k_mean - 1:
        return do_thuoc(x, C[s - 1], C[s], 100, F[s])
    return do_thuoc(x, C[s-1], C[s], C[s+1], F[s])

def predict(t, r = Rule_set):
    rule_burn = [min( [do_thuoc_set(float(t[i+1]), rule[i+1], Cluster_center[i], Fmc2[i]) for i in range(num_properties)]) for rule in r]
    do_thuoc_sum = [sum( [do_thuoc_set(float(t[i+1]), rule[i+1], Cluster_center[i], Fmc2[i])**0.5 for i in range(num_properties)]) for rule in r]
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
    # Predict class, Correct class, Rule_id
    return int(r[id][0]), int(t[0]), id, maxx, r[id][1:10], summ

train_correct = 0

def get_rule_truth(correct_class, t):
    sub_rules = []
    get_first = False
    first_id = 0
    for i, j in enumerate(Rule_set):
        if j[0] == correct_class:
            sub_rules.append(j)
            if not get_first:
                first_id = i
                get_first = True
    rule_id = predict(t, sub_rules)[2]
    d = predict(t, sub_rules)[3]
    r = predict(t, sub_rules)[4]
    return rule_id + first_id, d, r
# xx = 78
# b = predict(data[xx])[1]
# print(predict(data[xx]))
# print(get_rule_truth(b, data[xx]))


result = []
def detect_cluster(a, b, data):
    global result
    for i in range(9):
        if a[i] != b[i]:
            result += [[i, a[i], do_thuoc_set(data[i], a[i], Cluster_center[i]), b[i], do_thuoc_set(data[i], b[i], Cluster_center[i])]]
    # sai truoc, dung sau


g = open("PredictFailData.csv", "w")
for (i, t) in enumerate(data):
    if predict(t)[0] == predict(t)[1]:
        train_correct += 1
    else:
     
        b = predict(data[i])[1]
        x = predict(data[i])[4]
        y = get_rule_truth(b, data[i])[2]
        detect_cluster(x, y, t[1:10])
        g.write(str(i) + ',')
        # f.write("ban ghi {} \n".format(i))
        # # print(detect_cluster(x, y, t[1:10]))
        # f.write(",".join(str(i) for i in x) + "\n")
        # f.write(",".join(str(i) for i in y) + "\n")
        #     u1 = predict(data[i])[3]
        #     u2 = get_rule_truth(b, data[i])[1]
        #     print("New_c", (1-u2)/(u2 - 0.001) )
        #     print(i, "---------------------------")
def myF(a):
    return a[0]
result.sort(key=myF)
f = open("Edit2.csv", "w")
for i in result:
    f.write(",".join(str(j) for j in i) + "\n")


print("Kết quả cho tập Train")
print( "Đúng",train_correct,"trên",data_size,"dữ liệu")
print( "Độ chính xác: ", train_correct/data_size)      
print('--------------------------\n')  
# print("16,48,49,55,57,58,65,66,67,82,87,88,91,95,100,102,104,106,107,109,110,111,112,113,115,121,123")

# test_path = 'TestData.csv'
# test = read_file(test_path)
# test_size = len(test)

# test_correct = 0
# for t in test:
#     if predict(t)[0] == predict(t)[1]:
#         test_correct += 1
# print("Kết quả cho tập Test")
# print( "Đúng",test_correct,"trên",test_size,"dữ liệu")
# print( "Độ chính xác: ", test_correct/test_size)
# print('--------------------------')   
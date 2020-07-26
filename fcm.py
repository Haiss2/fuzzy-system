from read_data import read_file
import config as cf

path = 'TrainingData.csv'
data = read_file(path)
data_size = len(data)
num_properties = len(data[0]) - 1
k_mean = cf.k_mean
m_fuzzy = cf.m_fuzzy
e_fcm = cf.e_fcm
e_rule = cf.e_rule

Cluster_center = []

def step1():
    o = [0 for i in range(k_mean)]
    c = [0 + i*100/k_mean for i in range(k_mean)]
    u  = [ o.copy() for i in range(150)]
    return o, c, u

def step2(x, c, u):
    power = 2/(m_fuzzy - 1)
    for i in range(data_size):
        for j in range(k_mean):
            xi = float(data[i][x])
            mau = sum([(abs(xi - c[j]) / abs( xi - c[k]) )** power for k in range(k_mean)])
            u[i][j] = 1/mau

def step3(x, c, u):
    for j in range(k_mean):
        tu = sum([u[i][j]**m_fuzzy * float(data[i][x]) for i in range(data_size)])
        mau  = sum([u[i][j]**m_fuzzy for i in range(data_size)])
        c[j] = tu/mau

def step4(old, new):
    result = True
    for i in range(k_mean):
        result = result and abs(old[i] - new[i]) < e_fcm
    return result

for x in range(num_properties):
    o, c, u = step1()
    while not step4(o, c):
        o = c.copy()
        step2(x+1, c, u)
        step3(x+1, c, u)
    c.sort()
    Cluster_center.append(c)

# print("Các tâm cụm:", Cluster_center)

def do_thuoc(x, c1, c2, c3):
    if x >= c3 or x <= c1: 
        return 0
    if c1 < x and x <= c2:
        if c1 == 0:
            return 1
        return (x - c1) / (c2 - c1)
    if c2 < x and x < c3:
        if c3 == 100:
            return 1
        return (c3 - x) / (c3 - c2)

def burn(x, c):
    for id, pt in enumerate(c + [100]):
        if float(x) <= pt:
            break
    if id > 0 and id < len(c) - 1:
        u = do_thuoc(float(x), c[id - 1], c[id], c[id+1])
        return [(id - 1, 1 - u), (id, u)]
    if id == 0:
        return [(id, do_thuoc(float(x), 0, c[0], c[1] )), (0, 0)]
    if id == len(c):
        return [(id - 1, do_thuoc(float(x), c[id - 2], c[id -1], 100)), (0, 0)]
    if id == len(c) - 1:
        u = do_thuoc(float(x), c[id - 1], c[id], 100)
        return [(id - 1, 1 - u), (id, u)]
# print(data[0][1], Cluster_center[0], burn(data[0][1], Cluster_center[0]))

Rule_set = []


def add_rule(rule):
    loai = False
    for r in Rule_set:
        if r[1:num_properties + 1] == rule[1:num_properties + 1]:
            loai = True
            if r[num_properties] < rule[num_properties]:
                r = rule

    if not loai:
        Rule_set.append(rule)
import time
now = time.time()
for d in data:
    for i in range(2**num_properties):
        s = []
        b = '0'*(num_properties - len(bin(i)[2:])) + bin(i)[2:]
        u_ = 1
        for j in range(num_properties):
            burn_set = burn(d[j+1], Cluster_center[j])[int(b[j])]
            if u_ < e_rule:
                break
            s.append(burn_set[0])
            u_ *= burn_set[1]
        if u_ > e_rule:
            add_rule([d[0]] + s + [u_])
print("Tạo "+ str(len(Rule_set)) + " luật trong {:.2f}".format(time.time() - now) + 's')    


# predict
def do_thuoc_set(x, s, C):
    if s == 0:
        return do_thuoc(x, 0, C[0], C[1])
    if s == k_mean - 1:
        return do_thuoc(x, C[s - 1], C[s], 100)
    return do_thuoc(x, C[s-1], C[s], C[s+1])

def predict(t):
    rule_burn = [min( [do_thuoc_set(float(t[i+1]), rule[i+1], Cluster_center[i]) for i in range(num_properties)]) for rule in Rule_set]
    id = rule_burn.index(max(rule_burn))
    return Rule_set[id][0] ==t[0]

        

test_path = 'TestData.csv'
test = read_file(test_path)
test_size = len(test)

test_correct = 0
for t in test:
    if predict(t):
        test_correct += 1
print("Ket qua cho tap test")
print( test_correct, test_correct/test_size)

train_correct = 0
for t in data:
    if predict(t):
        train_correct += 1
print("Ket qua cho tap train")        
print( train_correct, train_correct/data_size)
    
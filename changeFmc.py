from read_data import read_file
import config as cf
import time

fmc_path = 'Fmc0.csv'
Fmc = read_file(fmc_path, 'float')

edit_path = 'Edit.csv'
Edit = read_file(edit_path, 'float')

for attr, flase, u_fasle, true, u_true in Edit:
    if true > flase:
        Fmc[attr][2*true] = (u_fasle+0.0001)/u_true
    else:
        Fmc[attr][2*true + 1] = (u_fasle+0.0001)/u_true

f = open(fmc_path, "w")
for i in Fmc:
    f.write(",".join(str(j) for j in i) + "\n")
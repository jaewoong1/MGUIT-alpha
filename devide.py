import os


txt = open('../sunny_train.txt', 'r').readlines()
n=0
for i in range(len(txt)):
    a = txt[i]
    if i == 0 :
        f = open('../test/img%d.txt' %n, 'w')
    else:
        if a[0] == '#':
            f.close()
            n+=1
            f = open('../test/img%d.txt' %n, 'w')
        else:
            f.write(a)

files_binary = ['a','b','c','d','e','f','h']
files_multi = ['a','b','c','d','e', 'h']

f = open('ans_binary.csv', 'r')
actual_binary = f.readlines()
f.close()

f = open('ans_multi.csv', 'r')
actual_multi = f.readlines()
f.close()

for x in files_binary:
    name = 'test_31'+x+'.csv'
    f = open(name, 'r')
    pred = f.readlines()
    f.close()
    acc = 0
    for i in range(18):
        if(pred[i] == actual_binary[i]):
            acc+=1
    print(name, ':', acc, '/ 18')

print()
for x in files_multi:
    name = 'test_32'+x+'.csv'
    f = open(name, 'r')
    pred = f.readlines()
    f.close()
    acc = 0
    for i in range(18):
        if(pred[i] == actual_multi[i]):
            acc+=1
    print(name, ':', acc, '/ 18')





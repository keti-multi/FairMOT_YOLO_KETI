# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



A=set("반팔티,반팔티,반팔티,반팔티,반팔카라티,반팔티,긴팔티,반팔티,반팔카라티,반팔카라티,반팔카라티,반팔티,반팔티,반팔티,반팔셔츠,반팔셔츠,반팔셔츠,반팔카라티,반팔티,반팔카라티,반팔티,린넨셔츠,반팔셔츠,반팔셔츠,반팔티,반팔카라티,반팔티,반팔티,반팔티,반팔티,반팔티,반팔티,반팔티,반팔카라티,긴팔티,반팔티,반팔카라티,반팔티,반팔티,반팔티,반팔티,반팔카라티,긴팔티,반팔티,반팔카라티".split(","))
B=set("")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
file = open('/media/syh/ssd2/data/DeepFashion/list_eval_partition.txt', "r") # deep fashion label 활용
top_set=set()
while True:
    line = file.readline()
    if len(line) < 38:
        pass
    else:
        print("line.split('                           ')[0].split('/')[2] : ",line.split('                           ')[0].split('/')[2])
        inn_set = set([str(line.split('                           ')[0].split('/')[2])])
        print("inn_set : ",inn_set)
        top_set=top_set.union(inn_set)
        print("top_set : ",top_set)
    if not line:
        break
        #print(line)

print(top_set)
print("len_top : ",len(top_set))
file.close()
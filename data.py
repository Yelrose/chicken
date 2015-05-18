import pandas as pd
from count import load_enrollment

dropout_map = {}
f = open("../train/truth_train.csv")
lines = f.readlines()
for li in lines:
    li = li.strip().split(",")
    li[0] = int(li[0])
    li[1] = int(li[1])
    dropout_map[li[0]] = li[1]


train_csv = pd.read_csv("../train/enrollment_train.csv",sep=",")
test_csv = pd.read_csv("../test/enrollment_test.csv",sep=",")
#course = set(list(train_csv["username"].values))
#course2 = set(list(test_csv["username"].values))
#print len(course),len(course2)
#print len(course | course2)
#print len(course & course2)


course = list(train_csv["course_id"].values)
course_dropout_rate = {}
for i in xrange(len(course)):
    if course[i] not in course_dropout_rate:
        course_dropout_rate[course[i]] = [0,0]
    drop = dropout_map[train_csv["enrollment_id"][i]]
    course_dropout_rate[course[i]][1] += 1
    course_dropout_rate[course[i]][0] += drop

for i,key in course_dropout_rate.items():
    print i , 1.0 * key[0]/key[1], key[0],key[1]


result = open("../result_event.txt","w")
item = list(test_csv["course_id"].values)
threshold = 90
enrollment = load_enrollment('../test/log_test.csv')
count = 0
for i in xrange(len(item)):
    idx = test_csv["enrollment_id"][i]
    if item[i] in course_dropout_rate:
        rate = course_dropout_rate[item[i]]
    else :
        rate = [1,1]
    rate = 1.0 * rate[0] / rate[1]
    if (enrollment[idx] < threshold and rate < 0.5) or (enrollment[idx] > threshold and rate > 0.5):
        count += 1
        rate = 1 - rate
    result.write("%s,%s\n" % (idx,rate) )
result.close()
print count




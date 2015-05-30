import pandas as pd








test = pd.read_csv("./test/enrollment_test.csv")
courses = set(test["course_id"].values)
sz = test["course_id"].size
dic = {}
course_count = {}
for i in courses:
    fp = open("./libfm_data/" + i)
    vec = fp.readlines()
    vec = [float(wd.strip()) for wd in vec]
    dic[i] = vec
    course_count[i] = 0

fp = open("result.csv","w")
for i in xrange(sz):
    course_id = test["course_id"][i]
    eid = test["enrollment_id"][i]
    ans = dic[course_id][course_count[course_id]]
    course_count[course_id] += 1
    fp.write("%s,%s\n" %(eid,str(ans)))
fp.close()







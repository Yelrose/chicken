import pandas as pd
import csv
import numpy as np



if __name__ == "__main__":
    object_csv = pd.read_csv("./object.csv")
    sz = object_csv["course_id"].size
    mark = {}
    for i in xrange(sz):
        course_id = object_csv["course_id"][i]
        if course_id not in mark:
            mark[course_id] = {}
        if object_csv["category"][i] != "problem":continue
        problem_id = object_csv["module_id"][i]
        if problem_id in mark[course_id]: continue
        mark[course_id][problem_id] = len(mark[course_id])


    enrollment_train = pd.read_csv("./train/enrollment_train.csv")
    enrollment_test = pd.read_csv("./test/enrollment_test.csv")

    map_enrollment_course = {}
    sz = enrollment_train["enrollment_id"].size
    for i in xrange(sz):
        enrollment_id = enrollment_train["enrollment_id"][i]
        course_id = enrollment_train["course_id"][i]
        map_enrollment_course[enrollment_id] = course_id


    sz = enrollment_test["enrollment_id"].size
    for i in xrange(sz):
        enrollment_id = enrollment_test["enrollment_id"][i]
        course_id = enrollment_test["course_id"][i]
        map_enrollment_course[enrollment_id] = course_id


    log_train = pd.read_csv("./train/log_train.csv")
    log_test = pd.read_csv("./test/log_test.csv")


    dic = {}
    sz = log_train["enrollment_id"].size
    for i in xrange(sz):
        #event object
        enrollment_id = log_train["enrollment_id"][i]
        course_id = map_enrollment_course[enrollment_id]
        if enrollment_id not in dic:
            dic[enrollment_id] = np.zeros(len(mark[course_id]),dtype="int")
        if log_train["event"][i] != "problem": continue
        problem_id = log_train["object"][i]
        if problem_id not in mark[course_id]:continue
        print enrollment_id,problem_id
        dic[enrollment_id][mark[course_id][problem_id]] = 1


    sz = log_test["enrollment_id"].size
    for i in xrange(sz):
        #event object
        enrollment_id = log_test["enrollment_id"][i]
        course_id = map_enrollment_course[enrollment_id]
        if enrollment_id not in dic:
            dic[enrollment_id] = np.zeros(len(mark[course_id]),dtype="int")
        if log_test["event"][i] != "problem": continue
        problem_id = log_test["object"][i]
        if problem_id not in mark[course_id]:continue
        print enrollment_id,problem_id
        dic[enrollment_id][mark[course_id][problem_id]] = 1


    writer = csv.writer(file('problem_vector.csv','wb'))
    writer.writerow(["enrollment_id","problem_vector"])
    for key,value in dic.items():
        value = list(value)
        value = [str(wd) for wd in value]
        value = " ".join(value)
        print key,value
        writer.writerow([key,value])







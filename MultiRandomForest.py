import pandas as pd
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

def load_label(path):
    dropout_map = {}
    f = open("./train/truth_train.csv")
    lines = f.readlines()
    for li in lines:
        li = li.strip().split(",")
        li[0] = int(li[0])
        li[1] = float(li[1])
        dropout_map[li[0]] = li[1]
    return dropout_map




def output_ans(path,ans,enrollment_train_csv):
    sz = enrollment_train_csv["enrollment_id"].size
    fp = open(path,"w")
    for i in xrange(sz):
        eid = enrollment_train_csv["enrollment_id"][i]
        cl = ans[i]
        fp.write("%s,%s\n" % (eid,cl))
    fp.close()




def even_count(enrollment_train_csv,log_train_csv):
    dic = {}
    for i in xrange(log_train_csv["enrollment_id"].size):
        idx = log_train_csv["enrollment_id"][i]
        if idx not in dic:
            dic[idx] = 0
        dic[idx] += 1
    return dic

def video_count(enrollment_train_csv,log_train_csv):
    dic = {}
    for i in xrange(log_train_csv["enrollment_id"].size):
        idx = log_train_csv["enrollment_id"][i]
        if idx not in dic:
            dic[idx] = 0
        if log_train_csv["event"][i] != "video":continue
        dic[idx] += 1
    return dic

def discussion_count(enrollment_train_csv,log_train_csv):
    dic = {}
    for i in xrange(log_train_csv["enrollment_id"].size):
        idx = log_train_csv["enrollment_id"][i]
        if idx not in dic:
            dic[idx] = 0
        if log_train_csv["event"][i] != "discussion":continue
        dic[idx] += 1
    return dic

def problem_count(enrollment_train_csv,log_train_csv):
    dic = {}
    for i in xrange(log_train_csv["enrollment_id"].size):
        idx = log_train_csv["enrollment_id"][i]
        if idx not in dic:
            dic[idx] = 0
        if log_train_csv["event"][i] != "problem":continue
        dic[idx] += 1
    return dic



def course_dropout(enrollment_train_csv,log_train_csv,train_truth):
    course = list(enrollment_train_csv["course_id"].values)
    course_dropout_rate = {}
    for i in xrange(len(course)):
        if course[i] not in course_dropout_rate:
            course_dropout_rate[course[i]] = [0,0]
        drop = train_truth[enrollment_train_csv["enrollment_id"][i]]
        course_dropout_rate[course[i]][1] += 1
        course_dropout_rate[course[i]][0] += drop
    for i,key in course_dropout_rate.items():
        course_dropout_rate[i] = 1.0* key[0]/key[1]
    return course_dropout_rate

def stu_dropout(enrollment_train_csv,log_train_csv):
    course = list(enrollment_train_csv["username"].values)
    course_dropout_rate = {}
    for i in xrange(len(course)):
        if course[i] not in course_dropout_rate:
            course_dropout_rate[course[i]] = [0,0]
        course_dropout_rate[course[i]][1] += 1
    for i,key in course_dropout_rate.items():
        course_dropout_rate[i] = 1.0* key[1]
    return course_dropout_rate



def Predicion(arg_dic):
    sz = arg_dic["enrollment_train_csv"]["enrollment_id"].size
    X = np.zeros((sz,6),dtype="float32")
    X = {}
    y = {}

    for i in xrange(sz):
        enrollment_id = arg_dic["enrollment_train_csv"]["enrollment_id"][i]
        course_id =  arg_dic["enrollment_train_csv"]["course_id"][i]
        if course_id not in X:
            X[course_id] = []
            y[course_id] = []
        username = arg_dic["enrollment_train_csv"]["username"][i]
        vec = np.zeros(6,dtype="float32")
        vec[0] = arg_dic["get_event_map"][enrollment_id]
        vec[1] = arg_dic["get_course_dropout_map"][course_id]
        vec[2] = arg_dic["get_stu_dropout_map"][username]
        vec[3] = arg_dic["get_video_count"][enrollment_id]
        vec[4] = arg_dic["get_discussion_count"][enrollment_id]
        vec[5] = arg_dic["get_problem_count"][enrollment_id]
        X[course_id].append(vec)
        y[course_id].append(arg_dic["truth_train"][enrollment_id])

    clf = {}
    for key,value in X.items():
        print "Training " + key
        clf[key] = GradientBoostingRegressor()
        clf[key] = clf[key].fit(X[key],y[key])

    sz = arg_dic["enrollment_test_csv"]["enrollment_id"].size
    res = []
    for i in xrange(sz):
        enrollment_id = arg_dic["enrollment_test_csv"]["enrollment_id"][i]
        course_id =  arg_dic["enrollment_test_csv"]["course_id"][i]
        username = arg_dic["enrollment_test_csv"]["username"][i]
        vec = np.zeros(6,dtype="float32")
        vec[0] = arg_dic["get_test_event_map"][enrollment_id]
        vec[1] = arg_dic["get_course_dropout_map"][course_id]
        vec[2] = arg_dic["get_test_stu_dropout_map"][username]
        vec[3] = arg_dic["get_test_video_count"][enrollment_id]
        vec[4] = arg_dic["get_test_discussion_count"][enrollment_id]
        vec[5] = arg_dic["get_test_problem_count"][enrollment_id]
        pro = clf[course_id].predict(vec)[0]
        if pro > 1: pro = 1
        if pro < 0: pro = 0
        res.append(pro)
    return res




if __name__  == "__main__":
    arg_dic = {}
    truth_train = load_label("./train/truth_train.csv")
    arg_dic["truth_train"] = truth_train



    enrollment_train_csv= pd.read_csv("./train/enrollment_train.csv")
    arg_dic["enrollment_train_csv"] = enrollment_train_csv
    enrollment_test_csv= pd.read_csv("./test/enrollment_test.csv")
    arg_dic["enrollment_test_csv"] = enrollment_test_csv
    log_train_csv = pd.read_csv("./train/log_train.csv")
    arg_dic["log_train_csv"] = log_train_csv
    log_test_csv = pd.read_csv("./test/log_test.csv")
    arg_dic["log_test_csv"] = log_test_csv

    get_event_map = even_count(enrollment_train_csv,log_train_csv)
    get_test_event_map = even_count(enrollment_test_csv,log_test_csv)

    arg_dic["get_event_map"] = get_event_map
    arg_dic["get_test_event_map"] = get_test_event_map

    get_video_count = video_count(enrollment_train_csv,log_train_csv)
    get_test_video_count = video_count(enrollment_test_csv,log_test_csv)

    arg_dic["get_video_count"] = get_video_count
    arg_dic["get_test_video_count"] = get_test_video_count



    get_discussion_count = discussion_count(enrollment_train_csv,log_train_csv)
    get_test_discussion_count = discussion_count(enrollment_test_csv,log_test_csv)

    arg_dic["get_discussion_count"] = get_discussion_count
    arg_dic["get_test_discussion_count"] = get_test_discussion_count


    get_problem_count = problem_count(enrollment_train_csv,log_train_csv)
    get_test_problem_count = problem_count(enrollment_test_csv,log_test_csv)

    arg_dic["get_problem_count"] = get_problem_count
    arg_dic["get_test_problem_count"] = get_test_problem_count


    get_course_dropout_map = course_dropout(enrollment_train_csv,log_train_csv,truth_train)
    arg_dic["get_course_dropout_map"] = get_course_dropout_map



    get_stu_dropout_map  = stu_dropout(enrollment_train_csv,log_train_csv)
    arg_dic["get_stu_dropout_map"] =  get_stu_dropout_map




    get_test_stu_dropout_map = stu_dropout(enrollment_test_csv,log_test_csv)
    arg_dic["get_test_stu_dropout_map"] = get_test_stu_dropout_map

    ans = Predicion(arg_dic)
    output_ans("./result.csv",ans,enrollment_test_csv)

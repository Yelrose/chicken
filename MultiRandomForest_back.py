import pandas as pd
import math
import numpy as np
import datetime
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression



def load_label(path):
    print sys._getframe().f_code.co_name
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
    print sys._getframe().f_code.co_name
    sz = enrollment_train_csv["enrollment_id"].size
    fp = open(path,"w")
    for i in xrange(sz):
        eid = enrollment_train_csv["enrollment_id"][i]
        cl = ans[i]
        fp.write("%s,%s\n" % (eid,cl))
    fp.close()




def even_count(enrollment_train_csv,log_train_csv):
    print sys._getframe().f_code.co_name
    dic = {}
    for i in xrange(log_train_csv["enrollment_id"].size):
        idx = log_train_csv["enrollment_id"][i]
        if idx not in dic:
            dic[idx] = 0
        dic[idx] += 1
    return dic

def video_count(enrollment_train_csv,log_train_csv):
    print sys._getframe().f_code.co_name
    dic = {}
    for i in xrange(log_train_csv["enrollment_id"].size):
        idx = log_train_csv["enrollment_id"][i]
        if idx not in dic:
            dic[idx] = 0
        if log_train_csv["event"][i] != "video":continue
        dic[idx] += 1
    return dic

def discussion_count(enrollment_train_csv,log_train_csv):
    print sys._getframe().f_code.co_name
    dic = {}
    for i in xrange(log_train_csv["enrollment_id"].size):
        idx = log_train_csv["enrollment_id"][i]
        if idx not in dic:
            dic[idx] = 0
        if log_train_csv["event"][i] != "discussion":continue
        dic[idx] += 1
    return dic

def problem_count(enrollment_train_csv,log_train_csv):
    print sys._getframe().f_code.co_name
    dic = {}
    for i in xrange(log_train_csv["enrollment_id"].size):
        idx = log_train_csv["enrollment_id"][i]
        if idx not in dic:
            dic[idx] = 0
        if log_train_csv["event"][i] != "problem":continue
        dic[idx] += 1
    return dic



def course_dropout(enrollment_train_csv,log_train_csv,train_truth):
    print sys._getframe().f_code.co_name
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


def course_vec(enrollment_train_csv,enrollment_test_csv):
    print sys._getframe().f_code.co_name
    dic = {}
    course_map = list(set(enrollment_train_csv["course_id"].values) | set(enrollment_test_csv["course_id"].values))
    countf = 0
    for i in course_map:
        dic[i] = countf
        countf += 1

    user_course = {}
    for i in xrange(enrollment_train_csv["course_id"].size):
        uid = enrollment_train_csv["username"][i]
        if uid not in user_course:
            user_course[uid] = np.zeros(39,dtype="float32")

        cid = enrollment_train_csv["course_id"][i]
        cid = dic[cid]
        user_course[uid][cid] = 1

    for i in xrange(enrollment_test_csv["course_id"].size):
        uid = enrollment_test_csv["username"][i]
        if uid not in user_course:
            user_course[uid] = np.zeros(39,dtype="float32")

        cid = enrollment_test_csv["course_id"][i]
        cid = dic[cid]
        user_course[uid][cid] = 1


    return user_course


def stu_dropout(enrollment_train_csv,log_train_csv):
    print sys._getframe().f_code.co_name
    course = list(enrollment_train_csv["username"].values)
    course_dropout_rate = {}
    for i in xrange(len(course)):
        if course[i] not in course_dropout_rate:
            course_dropout_rate[course[i]] = [0,0]
        course_dropout_rate[course[i]][1] += 1
    for i,key in course_dropout_rate.items():
        course_dropout_rate[i] = 1.0* key[1]
    return course_dropout_rate


def cross_course(enrollment_train_csv,enrollment_test_csv,log_train_csv,log_test_csv):
    print sys._getframe().f_code.co_name
    e_cross= []
    eid_2_uid = {}
    for i in xrange(enrollment_train_csv["enrollment_id"].size):
        eid = enrollment_train_csv["enrollment_id"][i]
        cid = enrollment_train_csv["username"][i]
        eid_2_uid[eid] = cid

    for i in xrange(enrollment_test_csv["enrollment_id"].size):
        eid = enrollment_test_csv["enrollment_id"][i]
        cid = enrollment_test_csv["username"][i]
        eid_2_uid[eid] = cid

    data = log_train_csv
    uid_date = {}
    for i in xrange(data["enrollment_id"].size):
        eid = data["enrollment_id"][i]
        uid = eid_2_uid[eid]
        t_ =  data["time"][i].split("T")[0]
        t = datetime.datetime.strptime(t_,"%Y-%m-%d")
        if len(e_cross) == 0 or e_cross[-1][0] != eid:
            if len(e_cross) != 0:
                if e_cross[-1][1] not in uid_date:
                    uid_date[e_cross[-1][1]] = []
                uid_date[e_cross[-1][1]].append((e_cross[-1][2],e_cross[-1][3]))
            e_cross.append((eid,uid,t,t))
        e_cross[-1] = (eid,uid,e_cross[-1][2],t)

    data = log_test_csv
    for i in xrange(data["enrollment_id"].size):
        eid = data["enrollment_id"][i]
        uid = eid_2_uid[eid]
        t_ =  data["time"][i].split("T")[0]
        t = datetime.datetime.strptime(t_,"%Y-%m-%d")
        if len(e_cross) == 0 or e_cross[-1][0] != eid:
            if len(e_cross) != 0:
                if e_cross[-1][1] not in uid_date:
                    uid_date[e_cross[-1][1]] = []
                uid_date[e_cross[-1][1]].append((e_cross[-1][2],e_cross[-1][3]))
            e_cross.append((eid,uid,t,t))
        e_cross[-1] = (eid,uid,e_cross[-1][2],t)
        if i == data["enrollment_id"].size - 1:
            if e_cross[-1][1] not in uid_date:
                uid_date[e_cross[-1][1]] = []
            uid_date[e_cross[-1][1]].append((e_cross[-1][2],e_cross[-1][3]))






    cross_course_map = {}
    for th in e_cross:
        uid = th[1]
        eid = th[0]
        date = (th[2],th[3])
        countf = 0
        for date1 in uid_date[uid]:
            if (date[0] <= date1[1] and date[0] >= date1[0]) or (date[1] <= date1[1] and date[1] >= date1[0]):
                countf += 1
        countf -= 1
        cross_course_map[eid] = countf
    return cross_course_map











def Predicion(arg_dic):
    print sys._getframe().f_code.co_name
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
        vec = np.zeros(7,dtype="float32")
        vec[0] = arg_dic["get_event_map"][enrollment_id]
        vec[1] = arg_dic["get_course_dropout_map"][course_id]
        vec[2] = arg_dic["get_stu_dropout_map"][username]
        vec[3] = arg_dic["get_video_count"][enrollment_id]
        vec[4] = arg_dic["get_discussion_count"][enrollment_id]
        vec[5] = arg_dic["get_problem_count"][enrollment_id]
        vec[6] = arg_dic["cross_course"][enrollment_id]
        vec = np.array(list(vec) + arg_dic["video_vector"][enrollment_id]+arg_dic["problem_vector"][enrollment_id] + arg_dic["cdatetime_vector"][enrollment_id] + arg_dic["datetime_vector"][enrollment_id] +arg_dic["rdatetime_vector"][enrollment_id]+ list(arg_dic["course_vec"][username]),dtype="float32")
        fp = open("./libfm_data/" + course_id + "_train.txt","a")
        fp.write(str(arg_dic["truth_train"][enrollment_id]))
        for i in xrange(len(vec)):
            fp.write(" " + str(i) +":" +str(vec[i]))
        fp.write("\n")
        fp.close()
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
        vec = np.zeros(7,dtype="float32")
        vec[0] = arg_dic["get_test_event_map"][enrollment_id]
        vec[1] = arg_dic["get_course_dropout_map"][course_id]
        vec[2] = arg_dic["get_test_stu_dropout_map"][username]
        vec[3] = arg_dic["get_test_video_count"][enrollment_id]
        vec[4] = arg_dic["get_test_discussion_count"][enrollment_id]
        vec[5] = arg_dic["get_test_problem_count"][enrollment_id]
        vec[6] = arg_dic["cross_course"][enrollment_id]
        vec = np.array(list(vec) + arg_dic["video_vector"][enrollment_id]+arg_dic["problem_vector"][enrollment_id] + arg_dic["cdatetime_vector"][enrollment_id] + arg_dic["datetime_vector"][enrollment_id] +arg_dic["rdatetime_vector"][enrollment_id]+ list(arg_dic["course_vec"][username]),dtype="float32")
        fp = open("./libfm_data/" + course_id + "_test.txt","a")
        fp.write("0")
        for i in xrange(len(vec)):
            fp.write(" " + str(i) +":" +str(vec[i]))
        fp.write("\n")
        fp.close()

        pro = clf[course_id].predict(vec)[0]
        if pro > 1: pro = 1
        if pro < 0: pro = 0
        res.append(pro )
    return res


def get_video_vector(path):
    print sys._getframe().f_code.co_name
    dic = {}
    data = pd.read_csv(path)
    sz = data["enrollment_id"].size
    for i in xrange(sz):
        eid = data["enrollment_id"][i]
        vec = data["video_vector"][i].split(" ")
        vec = [int(wd) for wd in vec]
        dic[eid] = vec
    return dic



def get_problem_vector(path):
    print sys._getframe().f_code.co_name
    dic = {}
    data = pd.read_csv(path)
    sz = data["enrollment_id"].size
    for i in xrange(sz):
        eid = data["enrollment_id"][i]
        vec = data["problem_vector"][i].split(" ")
        vec = [int(wd) for wd in vec]
        dic[eid] = vec
    return dic



def get_datetime_vector(path):
    print sys._getframe().f_code.co_name
    dic = {}
    data = pd.read_csv(path)
    sz = data["enrollment_id"].size
    for i in xrange(sz):
        eid = data["enrollment_id"][i]
        vec = data["datetime_vector"][i].split(" ")
        vec = [int(wd) for wd in vec]
        dic[eid] = vec
    return dic


def get_cdatetime_vector(path):
    print sys._getframe().f_code.co_name
    dic = {}
    data = pd.read_csv(path)
    sz = data["enrollment_id"].size
    for i in xrange(sz):
        eid = data["enrollment_id"][i]
        vec = data["cdatetime_vector"][i].split(" ")
        vec = [int(wd) for wd in vec]
        dic[eid] = vec
    return dic







def get_rdatetime_vector(path):
    print sys._getframe().f_code.co_name
    dic = {}
    data = pd.read_csv(path)
    sz = data["enrollment_id"].size
    for i in xrange(sz):
        eid = data["enrollment_id"][i]
        vec = data["rdatetime_vector"][i].split(" ")
        vec = [int(wd) for wd in vec]
        dic[eid] = vec
    return dic






if __name__  == "__main__":
    arg_dic = {}
    truth_train = load_label("./train/truth_train.csv")
    arg_dic["truth_train"] = truth_train


    arg_dic["video_vector"] = get_video_vector("./video_vector.csv")
    arg_dic["problem_vector"] = get_problem_vector("./problem_vector.csv")
    arg_dic["datetime_vector"] = get_datetime_vector("./datetime_vector.csv")
    arg_dic["rdatetime_vector"] = get_rdatetime_vector("./rdatetime_vector.csv")
    arg_dic["cdatetime_vector"] = get_cdatetime_vector("./cdatetime_vector.csv")



    enrollment_train_csv= pd.read_csv("./train/enrollment_train.csv")
    arg_dic["enrollment_train_csv"] = enrollment_train_csv
    enrollment_test_csv= pd.read_csv("./test/enrollment_test.csv")
    arg_dic["enrollment_test_csv"] = enrollment_test_csv
    log_train_csv = pd.read_csv("./train/log_train.csv")
    arg_dic["log_train_csv"] = log_train_csv
    log_test_csv = pd.read_csv("./test/log_test.csv")
    arg_dic["log_test_csv"] = log_test_csv


    arg_dic["cross_course"] = cross_course(enrollment_train_csv,enrollment_test_csv,log_train_csv,log_test_csv)

    arg_dic["course_vec"] = course_vec(enrollment_train_csv,enrollment_test_csv)

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

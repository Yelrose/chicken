import datetime
import pandas as pd
import numpy as np
import csv

if __name__ == "__main__":
    dic = {}
    data =  pd.read_csv("train/log_train.csv")
    sz = data["time"].size
    for i in xrange(sz):
        eid = data["enrollment_id"][i]
        t_ =  data["time"][i].split("T")[0]
        t = datetime.datetime.strptime(t_,"%Y-%m-%d")
        if eid not in dic:
            dic[eid] = {}
            dic[eid]["timeline"] = np.zeros(30,dtype="int")
            dic[eid]["min"] = t
        after = (t - dic[eid]["min"]).days
        dic[eid]["timeline"][after] = 1

    data =  pd.read_csv("test/log_test.csv")
    sz = data["time"].size
    for i in xrange(sz):
        i = sz - i - 1
        eid = data["enrollment_id"][i]
        t_ =  data["time"][i].split("T")[0]
        t = datetime.datetime.strptime(t_,"%Y-%m-%d")
        if eid not in dic:
            dic[eid] = {}
            dic[eid]["timeline"] = np.zeros(30,dtype="int")
            dic[eid]["min"] = t
        after = (dic[eid]["min"] - t).days
        dic[eid]["timeline"][after] = 1

    writer = csv.writer(file('rdatetime_vector.csv','wb'))
    writer.writerow(["enrollment_id","rdatetime_vector"])
    for key,value in dic.items():
        value = value["timeline"]
        value = list(value)
        value = [str(wd) for wd in value]
        value = " ".join(value)
        print key,value
        writer.writerow([key,value])


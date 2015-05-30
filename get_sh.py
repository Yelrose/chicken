import pandas as pd



cs = pd.read_csv("./train/enrollment_train.csv")


t = set(list(cs["course_id"].values))
t = list(t)



fp = open("do_all.sh","w")


for i in t:
    train_p = "./libfm_data/" + i +"_train.txt"
    test_p = "./libfm_data/" + i + "_test.txt"
    out_p = "./libfm_data/"  + i
    command = "~/libfm/bin/libFM -train %s -test %s -out %s -task r -iter 10000" % (train_p,test_p,out_p)
    fp.write(command +"\n")


fp.close()

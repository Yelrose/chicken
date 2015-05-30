import pandas as pd

class Node:

    def __init__(self,one_node=None,zero_node=None):
        self.l_child = one_node
        self.r_child = zero_node
        self.true_count = 0
        self.tot_count = 0

    def add_path(self,path,pos,label):
        self.tot_count += 1
        self.true_count += label
        pos += 1
        if pos == len(path):
            return
        one = path[pos]
        if one:
            if self.l_child == None:
                self.l_child = Node()
            self.l_child.add_path(path,pos,label)

        else :
            if self.r_child == None:
                self.r_child = Node()
            self.r_child.add_path(path,pos,label)

    def travel(self,out_str):
        print out_str,1.0 * self.true_count/self.tot_count,self.true_count,self.tot_count
        if self.l_child != None:
            self.l_child.travel(out_str + "1")
        if self.r_child != None:
            self.r_child.travel(out_str + "0")






class Tree:

    def __init__(self):
        self.head = Node()

    def add_path(self,path,label):
        self.head.add_path(path,-1,label)


    def travel(self):
        self.head.travel("")


if __name__ == "__main__":
    path_tree = Tree()
    rdate = pd.read_csv("./rdatetime_vector.csv")
    fp = open("./train/truth_train.csv")
    truth_log = {}
    for line in fp.readlines():
        line = line.strip().split(",")
        name = int(line[0])
        value = int(line[1])
        truth_log[name] = value

    for i in xrange(rdate["enrollment_id"].size):
        eid = rdate["enrollment_id"][i]
        vec = rdate["rdatetime_vector"][i].split(" ")
        vec = [int(wd) for wd in vec]
        if eid in truth_log:
            label = truth_log[eid]
            path_tree.add_path(vec,label)

    path_tree.travel()








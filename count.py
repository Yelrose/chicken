def load_enrollment(input_file):
    enrollment = {}
    with open(input_file) as f:
        f.readline()
        for line in f:
            arr = line.strip().split(',')
            arr[0] = int(arr[0])
            if arr[0] not in enrollment:
                enrollment[arr[0]] = 0
            enrollment[arr[0]] += 1
    return enrollment

def main():
    enrollment = load_enrollment('../train/log_train.csv')
    dropout_map = {}
    f = open("../train/truth_train.csv")
    lines = f.readlines()
    for li in lines:
        li = li.strip().split(",")
        li[0] = int(li[0])
        li[1] = int(li[1])
        dropout_map[li[0]] = li[1]

    zero = []
    one = []
    for each in enrollment:
        if dropout_map[each] == 0:
            zero.append(enrollment[each])
            #print each, enrollment[each], dropout_map[each]
        else:
            one.append(enrollment[each])
    len_0 = len(zero) / 2
    len_1 = len(one) / 2
    print sum(zero) * 1.0 / len(zero)
    print sum(one) * 1.0 / len(one)
    print max(zero), min(zero)
    print max(one), min(one)
    sord_0 = sorted(zero)
    print sord_0[len_0 - 1 : len_0 + 1], '\n', sord_0[-100:]
    sord_1 = sorted(one)
    print sord_1[len_1 - 1 : len_1 + 1], '\n',  sord_1[-100:]
    threshold = 125
    f_0 = filter(lambda x : x > threshold, sord_0)
    f_1 = filter(lambda x : x < threshold, sord_1)
    print len(f_0), len(zero)
    print len(f_1), len(one)
    print 'error num: %d' % (len(zero) - len(f_0) + len(one) - len(f_1))
if __name__ == '__main__':
    main()

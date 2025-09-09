import os, sys
import math
import matplotlib.pyplot as plt
import pandas as pd

#def dump_(_out, _tab, _dump):
#	cmd = "mcxdump -icl batch_MCL_out/%s -tabr %s -o batch_MCL_out/%s > /dev/null 2>&1" % (_out, _tab, _dump)
#	os.system(cmd)


def read_(_out):
    data = []
    with open(_out) as f:
        for line in f:
            c_i = line.strip().split()
            data.append(c_i)
    return data


def cal_mcs(_out):
    #dump_(_out, _tab, _dump)
    clusters = read_(_out)
    c = [len(x) for x in clusters]
    n = sum(c)
    try:
        mcs = sum([c_i * c_i for c_i in c]) / float(n * n)
    except ZeroDivisionError:
        print("Error: %s" % _out)
    print("# of clusters: %s" % len(clusters))
    print("# of nodes: %s" % n)
    print("Mean Cluster Size: %s" % mcs)
    return mcs


def main(file_base):
    results = []
    file_base = file_base
    #_tab = file_base + ".tab"
    for i in range(10, 150):
        print("Inflation: %s" % str(i/10))
        _out = "batch_MCL_out/out." + file_base + ".I" + str(i)
        #_dump = "dump." + file_base + ".I" + str(i)
        mcs = cal_mcs(_out)
        results.append((i/10, mcs))
    
    column = ['i', 'mcs']  # Column names for the dataframe
    df_results = pd.DataFrame(columns=column, data=results)
    df_results['d'] = abs(df_results['mcs'] - df_results['mcs'].shift(1))
    df_results.to_csv("mcs1-15.csv", index=None)

    x_, y_ = zip(*results)
    plt.plot(x_, y_, marker='D')
    plt.xticks(range(1, 15, 1))
    #plt.yticks(range(0, 3200, 200))
    plt.xlabel('Threshold main inflation value in MCL')  # x-axis label
    plt.ylabel('Mean cluster size')  # y-axis label
    #plt.title('Platform period 1-15')
   
    plt.savefig(r'MCL.png')
    # plt.show()


if __name__ == "__main__":
    file_base = sys.argv[1]
    main(file_base)

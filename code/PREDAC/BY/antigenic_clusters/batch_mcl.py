import os, sys
from decimal import Decimal
import time

start = time.time()
def mcl(_in, _out, inflation):
	cmd = "mcl %s -te 30 --abc -force-connected y -I %s -o batch_MCL_out/%s > /dev/null" % (_in, inflation, _out)
	os.system(cmd)

def main(infile, start, end):
    _in = infile
    for i in range(end):
        inflation = start  + i * Decimal('0.1')
        _out = "out.lit2020.I%s" % int(inflation * 10)
        print("Task: ", _in, _out, inflation)
        mcl(_in, _out, inflation)


if __name__ == "__main__":
    infile, start, end = sys.argv[1], Decimal(sys.argv[2]), int(sys.argv[3])
    main(infile, start, end)

end = time.time()
# print("The runing time of this process is {}h".format((end-start)/3600))
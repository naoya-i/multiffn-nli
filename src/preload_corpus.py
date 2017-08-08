import sys
import ioutils
import cPickle

print "Reading corpus from", sys.argv[1]

useful_data = ioutils.read_corpus(sys.argv[1], False)

print "Writing..."

with open(sys.argv[2], "wb") as f:
    cPickle.dump(useful_data, f)



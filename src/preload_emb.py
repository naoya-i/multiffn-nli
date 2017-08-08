import sys
import ioutils
import numpy as np

print "Loading embeddings from", sys.argv[1]
words, embeddings = ioutils.load_text_embeddings(sys.argv[1])

print "Saving numpy array..."
np.save(sys.argv[2], embeddings)

print "Saving vocab..."
with open(sys.argv[3], "w") as f:
    for w in words:
        print >>f, w


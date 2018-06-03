#!/usr/bin/env python

import glob
import sys
import os

if len(sys.argv) != 3:
    print "Usage: createDataset.py datasetName \"regEx\""
    sys.exit(1)
imagePath = sys.argv[2]
files = glob.glob(imagePath)

if not files:
    print "No data found!"
    sys.exit(1)

print "Found from {}".format(files[0])
print "Until {}".format(files[-1])

files = sorted(files)
datasetName = str(sys.argv[1]).replace(".dataset", "") + ".dataset"
files = [os.path.abspath(x) for x in files]
kittiPath = os.path.abspath(os.environ('KITTIPATH'))
files = [x.replace(kittiPath, '$KITTIPATH/') for x in files]

with open(datasetName, "w") as fp:
    fp.write("\n".join(files))
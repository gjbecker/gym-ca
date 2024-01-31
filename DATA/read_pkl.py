import pickle
import os
import sys
import pprint

if len(sys.argv) < 2:
    sys.exit('Missing filename arg!')
else:
    data = sys.argv[1]

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), data)

file = open(path, 'rb')

stats = pickle.load(file)

print(stats)
import pickle
import os
import sys
import pprint
import pandas as pd

if len(sys.argv) < 2:
    sys.exit('Missing filename arg!')
else:
    data = sys.argv[1]

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), data)

# file = open(path, 'rb')
# pkl = pickle.load(file)

# pprint.pprint(pkl)

data = pd.read_pickle(path)
try:    
    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))
except:
    print('')

with open(path.split('.')[0] + '.txt', 'w+') as f:
     pprint.pprint(data, f, sort_dicts=False, width=10)
# load pickle file
import matplotlib.pyplot as plt
import matplotlib
# import matplotlib.axes._subplots

import pickle

# matplotlib 3.7.1

file = r"C:\Users\flopo\Downloads\Results\2\fig.pickle"

# file = r"C:\Users\flopo\Downloads\Results\2\107_205_1_angular_momentum_evolution.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)
data.show()
plt.show()
# Import necessary packages
import numpy as np
from matplotlib.pyplot import *

x = np.array([[1,0,-0.5,0],[0,1,0,-1.5],[0,0,1.5,0],[0,0,0,2.5]]) 

print(np.linalg.inv(x))
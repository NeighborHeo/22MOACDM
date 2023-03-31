# In[ ]:
# ** import package **
import os
import sys
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

from tqdm import tqdm
from datetime import timedelta

from _utils.customlogger import customlogger as CL

pd.set_option('display.max_colwidth', -1)    #각 컬럼 width 최대로 
pd.set_option('display.max_rows', 50)        # display 50개 까지 
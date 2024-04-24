import numpy as np
from sklearn.model_selection import KFold

x = ["a", "b", "c", "d", "e", "f", "g"]

kf = KFold(n_splits=3, shuffle=False, random_state=None)

for train_index, test_index in kf.split(x):
    print("Train:", train_index, "Test:", test_index)
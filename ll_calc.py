import re

import numpy as np

PATTERNT = r"test_ll: -([0-9.]+)"

arr = np.empty(5)

with open("ll.txt", "r") as f:
    text = f.read()

matches = re.findall(PATTERNT, text)

for i, match in enumerate(matches):
    arr[i] = float(match)

print(arr)
print(len(matches))
print(arr.mean())
print(arr.std())
print(f"{arr.mean()}")

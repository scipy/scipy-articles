import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

count = subprocess.check_output(
    'git log | grep -E ^Date | awk \'{print " : "$3" "$6}\' | uniq -c',
    shell=True
).decode('utf-8').split('\n')

count_month = [line for line in count if line.strip()]
for c in count_month:
    if len(c.split(':')) != 2:
        print("error:", c)


count_month = [line.split(' : ') for line in count_month]
count_month = [(int(cnt), date) for (cnt, date) in count_month]

cnt, dates = zip(*count_month)

## TODO: use Gaussian filter?
cnt = np.convolve(cnt, np.ones(3)/3, mode='same')

ts = pd.Series(cnt, index=dates)
ts.plot(linewidth=2)

plt.show()

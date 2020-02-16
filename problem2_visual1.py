import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

# Hyper Parameters
POINT_NUM = 20

# Read the csv file.
csv_reader = csv.reader(open('./data/passingevents.csv'))

# The first match.(First match and self passing only.)
passing_list = [row for row in csv_reader if row[1]
                == 'Huskies']
passing_cnt = len(passing_list)

# Set the x-range.
x = np.linspace(0, 100, POINT_NUM)
y = [[] for _ in range(POINT_NUM)]

for i in range(passing_cnt - 1):
    distance = ((float(passing_list[i][9]) - float(passing_list[i][7])) **
                2 + (float(passing_list[i][10]) - float(passing_list[i][8])) ** 2) ** 0.5
    distance += ((float(passing_list[i+1][7]) - float(passing_list[i][7])) **
                 2 + (float(passing_list[i + 1][8]) - float(passing_list[i][8])) ** 2) ** 0.5
    t = float(passing_list[i + 1][5]) - float(passing_list[i][5])

    if t <= 0.2:
        continue

    average_x = (float(
        passing_list[i][7]) + float(passing_list[i][9]) + float(passing_list[i + 1][7])) / 3

    y[int(average_x * POINT_NUM / 100)].append(distance / t)

y = np.array([(np.mean(np.array(i)) if len(i) > 0 else np.nan) for i in y])

plt.plot(x, y, color='blue', linewidth=1)
plt.xlabel('x-pos')
plt.ylabel('average speed')
plt.show()

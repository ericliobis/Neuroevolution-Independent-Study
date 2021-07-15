
import matplotlib.pyplot as plt
import csv
import numpy as np
plt.switch_backend('agg')
x = []
y = []
avg =0
avg_over = 10
with open('SCORE_HIST_THEIRCODE_2.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    
    for count, row in enumerate(lines):
        print(count)
        for count1, row1 in enumerate(row):
            print(count1)
            avg = avg+ float(row1)
            if count % avg_over == 0:
                x.append(int(count/avg_over))
                y.append(float(avg/avg_over))
                avg =0


plt.plot(x, y, color = 'g')
plt.title('Their Code', fontsize = 20)
plt.savefig('their code 2.png')
print("Plot Done")
plt.close()

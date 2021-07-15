import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random
import glob
import argparse
import statistics 
import csv
import os


parser = argparse.ArgumentParser()
parser.add_argument("file_loc", help="The Location of the folders to loop through")
args = parser.parse_args()


def plot_and_save(subdir,file):
    test = []
    gap = 10
    path = os.path.join(subdir, file)
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            test.append(line)

    test =  [list( map(int,map(float,i)) ) for i in test]
    test2 = test.copy()
    print(len(test2))
    print(len(test2[0]))
    #lists to keep track of the mean, lower/upper bound of error relative to mean
    mean_values=[]
    lower_errors=[]
    upper_errors=[]
    maxSize = 0
    currentConsider = []
    all_runs = []
    for i in test:
        if len(i) >maxSize:
            maxSize = len(i)

    for i in test:
        currentConsider.append(i[0:gap])

    for i in range(maxSize-gap):
        dummy = []
        for j in currentConsider:
            dummy.append(statistics.mean(j))
        mean = statistics.mean( dummy)
    #mean calculated and stored for plotting later
        mean_values.append( mean)
    #calculate the standard deviation
        sd = 0
        if len(dummy)>1:
            sd = statistics.stdev(dummy)
        lower_errors.append( mean - sd )
        upper_errors.append( mean + sd )
        toPop = []
        for count, j in enumerate(test):
            if i+gap< len(j):
                currentConsider[count].pop(0)
                currentConsider[count].append(j[i+gap])
            else:
                toPop.append(count)
        for i in reversed(sorted(toPop)):
            currentConsider.pop(i)
            test.pop(i)
     

            
        all_runs.append(dummy)

        

    print(len(test2))
    print(len(test2[0]))
    #simple produce a list with 0....to (generations-1) in it
    x = range(maxSize-gap)
    y = mean_values

    #draw mean line
    plt.plot(x, y, color="red", label="Average Score for 10 Runs" )
    #draw shaded region from lower to upper
    plt.fill_between(x, lower_errors, upper_errors, alpha=0.25, facecolor="red", label="standard deviation")
    plt.fill_between(x, 850, 1000, alpha=0.25, facecolor="blue", label="success region")
    #show a legend
    plt.legend()

    #add labels to the axes
    plt.xlabel('generation')
    plt.ylabel('score')

    #saves to file
    plt.savefig(os.path.join(subdir, '10_run_avg.png'))
    print("Plot Done")
    plt.close()
    n = len(test2)
    color=iter(cm.rainbow(np.linspace(0,1,n)))

    for i in test2:
        spliter = [i[x:x+gap] for x in range(0, len(i),gap)]
        avg = []
        for j in spliter:
            avg.append(statistics.mean(j))
        plt.plot(range(len(avg)), avg, color=next(color), label="10 Runs" )
    #draw shaded region from lower to upper
    #show a legend
    plt.legend()

    #add labels to the axes
    plt.xlabel('generation')
    plt.ylabel('score')

    #saves to file
    plt.savefig(os.path.join(subdir, '10_runs.png'))
    print("Plot Done")
    plt.close()
    #shows on the screen
    #plt.show()


for subdir, dirs, files in os.walk(args.file_loc):
    for file in files:
        if file == "SCORE_HIST_ALL.csv":
            plot_and_save(subdir,file)


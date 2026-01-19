import numpy as np

# this file is used to perform any manual processing of dataset files that I need to do

def find_avg_difference_in_column(path):
    # I am using this to find the average difference between samples timestamps.
    # I am using the values from this to generate the missing timestamps.
    differences = []
    previous_line = 0

    with open(path, "r") as f:
        lines = f.readlines()

        for line in lines:
            current_line = float(line)
            if previous_line == 0: # this will skip the first run so as it doesn't have a previous value to compare to.
                previous_line = current_line
                continue
            differences.append(current_line - previous_line)
            previous_line = current_line

    f.close()
    avg = np.mean(differences)
    print("Mean difference: " + str(avg))
    median = np.median(differences)
    print("Median difference: " + str(median)) # I will probably use this value as it is a bit more immune to outliers

def find_avg_difference_between_columns(Firstpath, Secondpath):
    # I am using this function to find the average difference between the Start and Stop Time.
    # I am using this to generate the missing Stop Time values
    firstFileLines = []
    secondFileLines = []
    firstFileValues = []
    secondFileValues = []

    with open(Firstpath, "r") as f:
        firstFileLines = f.readlines()
    f.close()
    with open(Secondpath, "r") as f2:
        secondFileLines = f2.readlines()
    f2.close()

    for line in firstFileLines:
        firstFileValues.append(float(line))

    for line in secondFileLines:
        secondFileValues.append(float(line))

    differnces = [] # I have deliberately left this misspelled to ensure the there is no confusion with the one in the other function.
    for i in range(len(firstFileValues)):
        differnces.append(secondFileValues[i] - firstFileValues[i])

    avg = np.mean(differnces)
    print("Mean difference: " + str(avg))
    median = np.median(differnces)
    print("Median difference: " + str(median))

if __name__ == '__main__':
    # These are the non-null values from the sessions_7_vDNS.csv that I have saved to my desktop.
    # This file is included in the GitHub to show how I came up with the values for the manual processing. They are not intended to be run by anyone but me.
    # I am not going to add these random text files to this repo.
    # If you want you can recreate these files yourself by copy-pasting the Start Time and Stop Time, lines 47-11892 into separate text files and changing the paths here to their locations.
    path1 = "/home/diarmaid/Desktop/start_time_values.txt"
    path2 = "/home/diarmaid/Desktop/stop_time_values.txt"
    print("differences in start time")
    find_avg_difference_in_column(path1)
    print("differences in stop time")
    find_avg_difference_in_column(path2)
    print("differences between start and stop time")
    find_avg_difference_between_columns(path1, path2)

# I got the following output from this code:
# differences in start time
# Mean difference: 32893.45588856057
# Median difference: 904.0
# differences in stop time
# Mean difference: 32893.70688054031
# Median difference: 1283.0
# differences between start and stop time
# Mean difference: 99939.99240249873
# Median difference: 100.0
#
# Process finished with exit code 0
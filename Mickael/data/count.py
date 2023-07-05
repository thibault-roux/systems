import numpy as np


# read test.csv file to count the length of each word


def count():
    lengths = []
    with open('test.csv', 'r') as f:
        next(f)
        for line in f:
            sent = line.strip().split(',')[-1]
            for word in sent.split(" "):
                lengths.append(len(word))

    lengths = np.array(lengths)
    return lengths

if __name__ == "__main__":
    lengths = count()
    print("The average length of words is: ", np.mean(lengths))
    print("The max length of words is: ", np.max(lengths))
    print("The min length of words is: ", np.min(lengths))
    print("The standard deviation of words is: ", np.std(lengths))
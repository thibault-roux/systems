import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot(files):
    # plot one file if len(files) == 1 and all files if len(files) > 1
    # create one plot for WER and one plot for CER
    WER = [] # list of lists
    CER = [] # list of lists
    for file in files:
        # read file containing lines like "epoch: 1, CER: 7.90, WER: 23.00"
        with open("data/" + file + ".txt", 'r') as f:
            lines = f.readlines()

        # extract CER and WER
        cer = []
        wer = []
        for line in lines:
            cer.append(float(line.split(',')[1].split(':')[1]))
            wer.append(float(line.split(',')[2].split(':')[1]))
        CER.append(cer)
        WER.append(wer)

    # plot CER
    for i in range(len(files)):
        cer = CER[i]
        file = files[i]
        plt.plot(cer, label=file)
    plt.title('CER')
    plt.xlabel('epoch')
    plt.ylabel('CER')
    plt.legend()
    plt.show()
    plt.savefig('figs/CER.png')

    plt.clf()

    # plot WER
    for i in range(len(files)):
        wer = WER[i]
        file = files[i]
        plt.plot(wer, label=file)
    plt.title('WER')
    plt.xlabel('epoch')
    plt.ylabel('WER')
    plt.legend()
    plt.show()
    plt.savefig('figs/WER.png')




if __name__ == '__main__':
    # parse arguments
    # receive one or many strings
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=str)
    
    # parse arguments
    args = parser.parse_args()
    files = args.files

    # read files
    plot(files)
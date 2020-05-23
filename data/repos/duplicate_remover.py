import os
import sys
import click
import pandas as pd

if __name__ == '__main__':
    file_name = 'repos_final2014-part2.csv'
    inFile = open(file_name, 'r')
    listLines = []
    for line in inFile:
        if line in listLines:
            print(line)
            continue
        else:
            listLines.append(line)
    inFile.close()

    outFile = open(file_name, 'w')
    for line in listLines:
        outFile.write(line)
    outFile.close()

import csv
import logging
import time
from datetime import datetime, timedelta
from string import Template

import requests
from nltk import sent_tokenize

from future.src.data_generation.calculator import Calculator

def extract_data(answer):
    is_OB_EB_S2R = False
    calc = Calculator(threshold=0)
    if calc.utility(1234, answer, '') > 0:
        is_OB_EB_S2R = True
    return is_OB_EB_S2R


if __name__ == '__main__':
    with open('data.txt', 'r') as file:
        data = file.read().replace('\n', '')
        print(extract_data(data))

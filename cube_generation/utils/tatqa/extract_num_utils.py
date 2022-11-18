# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import string
from typing import List
import numpy as np

EXCLUDE_IN_NUM = "'\"\\$€£¥(),[]"
def _clean_num(text:str):
    return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])

def scale_to_num(scale):
    scale = scale.lower()
    num = 1
    if 'hundred' in scale:  # hundred
        num = 100
    elif 'thousand' in scale:  # thousand
        num = 1000
    elif 'million' in scale:  # million
        num = 1000000
    elif 'billion' in scale:  # billion
        num = 1000000000
    elif 'percent' in scale:  # percent
        num = 0.01
    return num

def extract_one_num_from_str(s):
    s = _clean_num(s)
    r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
    groups = re.findall(r_num, s)
    if len(groups) == 0:
        return None
    num = groups[0][0]
    if num == '':
        return None
    if '.' in num:
        return float(num)
    return int(num)


def negative_num_handle(x):
    """
    :param x:  transform (134) -> -134
    :return:
    """
    all = re.findall('(\([\d.\s]+\))', x.strip())
    if len(all) > 0:
        return -1
    return 1

def percent_num_handle(x):
    """
    :param x:  transform 12% -> 12/100
    :return:
    """
    all = re.findall('([\d.\s]+%)', x.strip())
    if len(all) > 0:
        return 0.01 #FIXME #FIXME FIXME FIXME FIXME
    return 1

def word_scale_handle(x):
    """
    :param x: 1 million = 1,000,000
    :return:
    """
    iter = re.finditer('([\d.]+\s?[a-zA-Z]+)', x)
    for one in iter:
        text = one.group(0).lower()
        scale_val = scale_to_num(text)
        return scale_val
    return 1

def to_number(text:str) -> float:
    num = extract_one_num_from_str(text)
    scale_val = word_scale_handle(text)
    negative_flag = negative_num_handle(text)
    percent_flag = percent_num_handle(text)
    if num is not None:
        return round(num * scale_val * negative_flag * percent_flag, 4)
    return None

def facts_to_nums(facts):
    return [to_number(f) for f in facts]


def is_pure_number(text):
    cnt_ = 0
    num_ = to_number(text)
    if num_ is None:
        return False
    if num_ >= 0 and "-" in text:
        return False
    for t in text:
        if t.isalpha():
            cnt_ += 1
    if cnt_ >= 3:
        return False
    return True

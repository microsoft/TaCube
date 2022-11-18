# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

def _is_average(num_facts:list, answer):
    return round(np.average(num_facts), 2) == round(answer, 2)

def _is_change_ratio(num_facts:list, answer):
    if len(num_facts) != 2:
        return False
    cands = []
    if num_facts[1] != 0:
        ori_percent = round(100 * (num_facts[0] - num_facts[1]) / num_facts[1], 2)
        cands.append(ori_percent)
    if num_facts[0] != 0:
        ori_percent = round(100 * (num_facts[1] - num_facts[0]) / num_facts[0], 2)
        cands.append(ori_percent)
    return round(answer, 2) in cands

def _is_division(num_facts:list, answer):
    if len(num_facts) != 2:
        return False
    cands = []
    if num_facts[1] != 0:
        cands.append(round(num_facts[0]/num_facts[1], 2))
        cands.append(100 * round(num_facts[0]/num_facts[1], 2))
    if num_facts[0] != 0:
        cands.append(round(num_facts[1]/num_facts[0], 2))
        cands.append(100 * round(num_facts[1]/num_facts[0], 2))
    return round(answer, 2) in cands

def _is_diff(num_facts:list, answer):
    if len(num_facts) != 2:
        return False
    ans_1 = round(num_facts[0] - num_facts[1], 2)
    ans_2 = round(num_facts[1] - num_facts[0], 2)
    return round(answer, 2) in (ans_1, ans_2)

def _is_sum(num_facts:list, answer):
    return round(np.sum(num_facts), 2) == round(answer, 2)

def _is_times(num_facts:list, answer):
    return round(np.prod(num_facts), 2) == round(answer, 2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:19:06 2023

@author: konstantinos
"""

import os

def isalice():
    ''' Checks if we are in alice'''
    pwd = os.getcwd()
    key = 'data1' # NEVER NAME A LOCAL FOLDER data1
    flag = key in pwd
    if flag:
        alice = True
        plot = False
    else:
        alice = False
        plot = True
    return alice, plot

if __name__ == '__main__':
    alice, plot = isalice()
    print(alice)
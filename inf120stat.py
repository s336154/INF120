# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:11:50 2024

__author__= Sari Siraj Abdalla Ali
__email__ = sari.siraj.abdalla.ali@nmbu.com
"""

import math

def mean(list):
    sum = 0
    for i in list:
        sum += i
    return sum/len(list)



def std(list):
    m = mean(list)
    x= 0
    sum= 0 
    for i in list:
        x= (i-m)**2
        sum += x       
    return math.sqrt(sum/len(list))
    

l = [5, 5, 5, 3]
print(f"\nThe mean is: {mean(l):.2f}")
print(f"\nStandard deviation is: {std(l):.2f}")

        
    


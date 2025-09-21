# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:00:23 2024

@author: saris
"""
dict1 = {}
with open('julehandel.txt', 'rt') as file:
    lines = file.readlines()
    for line in lines[2:]:
        key = line.split("\t")[0]
        amount = line.split("\t")[1]
        price = line.split("\t")[2]
        payed = float(amount)*float(price)
        dict1[key]= payed

sorted_dict = dict(sorted(dict1.items()))
total = 0
print("-"*43)
for key, value in sorted_dict.items():
    total += value  
    print(f"{key:35} {value:.2f} kr")
print("-"*43)
print(f"{'Sum':35} {total:.1f} kr")

        

        
   

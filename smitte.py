# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:54:25 2024

__author__= Sari Siraj Abdalla Ali
__email__ = sari.siraj.abdalla.ali@nmbu.com
"""
import csv

def updateSIR(s_y, i_y, r_y, b, k):
    s_t = s_y - b*s_y*i_y
    i_t = i_y + b*s_y*i_y - k*i_y
    r_t = r_y + k*i_y          
    return s_t, i_t, r_t


# Startbetingelser
s_0 = 1.0  
i_0 = 10 / 5e6 
r_0 = 0.0  

b = 1 / 3  
k = 1 / 10 

days = 120
results = []

s_t, i_t, r_t = s_0, i_0, r_0


for day in range(days + 1):
    results.append([day, s_t, i_t, r_t])
    s_t, i_t, r_t = updateSIR(s_t, i_t, r_t, b, k)


# Skriv resultatene til CSV-filen pan.csv
with open("pan.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Day", "Susceptible", "Infectious", "Recovered"])
    writer.writerows(results)


#print(updateSIR(2, 3, 4, 1/3, 1/10))
            
    
    
            
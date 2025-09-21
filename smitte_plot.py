# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:41:14 2024

__author__= Sari Siraj Abdalla Ali
__email__ = sari.siraj.abdalla.ali@nmbu.com

"""

import matplotlib.pyplot as plt
import numpy as np


nestet = [[],[],[],[]]

with open('pan.csv', 'rt') as pan:
    file = pan.readlines()
    for line in file[1:]:
        nestet[0].append(float(line.split(",")[0]))
        nestet[1].append(float(line.split(",")[1]))
        nestet[2].append(float(line.split(",")[2]))
        nestet[3].append(float(line.split(",")[3]))
        
d_list = np.array(nestet[0])
s_list = np.array(nestet[1])
i_list = np.array(nestet[2])
r_list = np.array(nestet[3])

plt.figure(figsize=(10, 6))
plt.plot(d_list, s_list, label="Susceptible", color="blue", linestyle="-.")
plt.plot(d_list, i_list, label="Infectious", color="red", linestyle=":")
plt.plot(d_list, r_list, label="Recovered", color="green", linestyle="--")
plt.title("Spread of COVID-19")
plt.xlabel("Days")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()     

##################### SVAR PÅ SPØRSMÅL ###################

# 1. Omtrent 35% av befolkning var smittet samtidig
# 2. Etter 114 dager har de fleste blitt frisk igjen

##########################################################
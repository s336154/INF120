# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:05:29 2024

__author__= Sari Siraj Abdalla Ali
__email__ = sari.siraj.abdalla.ali@nmbu.com
"""

### DEL 1

import math

jord_radius_km = 6371
jord_radius_m = jord_radius_km * 1000

jord_volum_m = 4/3* math.pi * jord_radius_m**3

print("#" *50)
print(f'{"#" *21} DEL 1 {"#" *22} ')
print("#" *50)
print(f"\n{'Jordens volum er omtrent: ':>30}{jord_volum_m:.3e} m^3\n")

### DEL 2

jor_masse_kg = 5.9737 * 10**24
jord_massetetthet = jor_masse_kg/jord_volum_m

print("#" *50)
print(f'{"#" *21} DEL 2 {"#" *22} ')
print("#" *50)
print(f"\n{'Jordens massetetthet er omtrent: ':>35}{jord_massetetthet:.2f} kg/m^3\n")
print("#" *50)
print("#" *50)
print("#" *50)







# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:57:50 2024

__author__= Sari Siraj Abdalla Ali
__email__ = sari.siraj.abdalla.ali@nmbu.com
"""

navn = [
     ['Tore', 'Pettersen'],
     ['Nils', 'Olavsen'],
     ['Aase', 'Lund'],
     ['Kristine', 'Oremo'],
     ['Tina', 'Kittelsen'],
     ['Per', 'Carstensen'],
     ['Lena', 'Nilsen'],
     ['Karsten', 'Woll'],
     ['Ine', 'Ørstad'],
     ['Ravn', 'Havnås'],
     ['Navn', 'Navnesen'],
     ['Kari', 'Nordmann'],
     ['Lille', 'Marius'],
     ['Jesper', 'Danberg']]

condition = False


for i in range(len(navn)):
    if navn[i][0] == "T" or len(navn[i][1]) > 6 or (navn[i][0] == "Navn" and navn[i][1] == "Navnesen" ):
        condition = True
    if condition == True:
        print(f"{i+1} {navn[i][0]} {navn[i][1]}")
        condition = False
    
        
        
    
    
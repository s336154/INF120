# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:39:35 2024

__author__= Sari Siraj Abdalla Ali
__email__ = sari.siraj.abdalla.ali@nmbu.com
"""
#import numpy as np
from inf120stat import mean, std

def choose_action():
   print("-"*45 )
   print("\nVelkommen til Enkelt Statistikk Program (ESP)\n")
   print("-"*45 )
   print("\nESP Meny: ")
   print(f"{'1. Les inn verdier':>20}")
   print(f"{'2. Tøm verdi listen':>21}")
   print(f"{'3. Vis gjennomsnitt og standardavvik':>38}")
   print(f"{'4. List ut verdiene':>21}")
   print(f"{'5. Avslutt':>12}\n")
   inp = int(input("Ditt valg: "))
   
   while inp not in range(1,6):
       print("\nVennligst skriv inn et tall melllom 1 og 5.\n")
       inp = int(input("Ditt valg: "))
       
   return inp


def read_values():
    values = []
    inp = float(input("\nNy verdi (Enter = slutt): "))
    
    try:
        while isinstance(inp, float):
            values.append(inp)
            inp = float(input("\nNy verdi (Enter = slutt): "))
        
    except ValueError:
        print("Avsluttet innlesning\n\n")
    
    return values
        
 
def print_statistics(values):
    ant = len(values)
    m = mean(values)
    s = std(values)
    
    print(f"\nAntall verdier: {ant}, gjennomsnitt: {m:.2f}, standardavvik: {s:.2f}\n\n")
    
    
    
values = []
valg = choose_action()

while valg:
    if valg == 1:
        values = read_values()
        valg = choose_action()
        
    elif valg == 2:
        if values != []:
            values.clear()
        valg = choose_action()
       
    elif valg == 3:
        print_statistics(values)
        valg = choose_action()
        
    elif valg == 4:
        print("\n")
        
        if values != []:
            for i in values:
                print(i, end="   ")
        else:
            print("Ingen verdier er lest inn eller listen er tom.")          
       
        print("\n\n")
        valg = choose_action()
         
    else:
        break
    
       
   

   


#print(choose_action())
       
   
   


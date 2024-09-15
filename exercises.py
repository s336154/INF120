# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:33:44 2024

@author: saris
"""
"""
for i in range(9): print(i)


numbers = [12, 13, 14,] 
doubled = [x *2  for x in numbers] 
print(doubled)
lis = ("Even number" if i % 2 == 0 
       else "Odd number" for i in range(8))
print(lis) 

"""
"""

 Exercise 1.4: Convert from meters to British length units
 Makeaprogramwhereyouset alength givenin meters and then compute and write
 out the corresponding length measured in inches, in feet, in yards, and in miles. Use
 that one inch is 2.54cm, one foot is 12 inches, one yard is 3 feet, and one British
 mile is 1760yards. For verification, a length of 640 meters correspondsto 25196.85
 inches, 2099.74 feet, 699.91 yards, or 0.3977 miles.
 Filename: length_conversion
 


inp_meter = float(input("\n\nWhat is the measurement in meters: "))
meas_inches = inp_meter*100/(2.54)
meas_foot = meas_inches/12
meas_yard= meas_foot /3
out_txt = f"The mesurement {inp_meter} meters is: \n {meas_inches} inches \n {meas_foot} feet \n {meas_yard} yard \n  "
print(out_txt)

"""
"""



# Exercise 2.1

# Function to convert Fahrenheit to Celsius

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5.0 / 9.0

# Print the header
print(f"{'Fahrenheit':>10} | {'Celsius':>7}")
print("-" * 20)

# Loop to generate the table
for fahrenheit in range(0, 101, 10):
    celsius = fahrenheit_to_celsius(fahrenheit)
    print(f"{fahrenheit:>10} | {celsius:>7.2f}")




 Exercise 2.4: Generate odd numbers
 Write a program that generates all odd numbers from 1 to n.Setn in the beginning
 of the program and use a while loop to compute the numbers. (Make sure that if n
 is an even number, the largest generated odd number is n-1.)
 Filename: odd.
 
 Exercise 2.5: Compute the sum of the first n integers
 Write a program that computes the sum of the integers from 1 up to and including
 n. Compare the result with the famous formula n.n C 1/=2.
 Filename: sum_int.



#Exc 2.4
inp_n = int(input("\n\nEnter a number: "))

i=1;

while i<inp_n:
    if 1%2==1:
        print(f"{i}\n")
        if i == inp_n -1: print(f' Yes!! {i} is {inp_n} - 1 !')
        i+=1
       
 
# Exc 2.5

inp_n = int(input("\n\nEnter a number: "))
i=1;
sum = inp_n
while i< inp_n:
    sum += i
    i += 1
print(f"\nThe sum is {sum}")
formula_f = int(inp_n*(1+inp_n)/2)
print(f"\nThe famous formula says {formula_f} ")
    


#Exercise 2.7

list_n = [[n, n+1] for n in range(2,20)]
print(list_n)


# Parameters
a = 0
b = 10
n = 5  # number of intervals

# Calculate the interval length
h = (b - a) / n

# Initialize an empty list to store the coordinates
x_coords = []

# Use a for loop to generate the coordinates and append them to the list
for i in range(n + 1):
    xi = a + i * h
    x_coords.append(xi)

# Print the result
print("Coordinates using a for loop:", x_coords)



# Use a list comprehension to generate the coordinates
x_coords_comp = [a + i * h for i in range(n + 1)]

# Print the result
print("Coordinates using list comprehension:", x_coords_comp)



#Exercise 2.8

# Given values
v0 = 10.0  # Initial velocity in m/s
g = 9.81   # Acceleration due to gravity in m/s^2
n = 10     # Number of intervals

# Time interval
t_max = 2 * v0 / g
t_values = [t_max * i / n for i in range(n+1)]

print("Using for loop:")
print(f"{'t (s)':>10} {'y(t) (m)':>15}")
for t in t_values:
    y = v0 * t - 0.5 * g * t**2
    print(f"{t:>10.4f} {y:>15.4f}")

# Using a while loop
print("\nUsing while loop:")
print(f"{'t (s)':>10} {'y(t) (m)':>15}")

t = 0
i = 0
while t <= t_max:
    y = v0 * t - 0.5 * g * t**2
    print(f"{t:>10.4f} {y:>15.4f}")
    i += 1
    t = t_max * i / n



# RANDOM
a = [1, 3, 5, 7, 11]
b = [13, 17]
c = a + b
print(c)

eps = 1.0
i=0
while 1.0 != 1.0 + eps:
 print("...............", eps)
 eps = eps/2.0
 i += 1
print("final eps:", eps)
print("i = ", i)

x =1
print(x)
x =1.
print(x)
x =1;
print(x)
#x =1!
#print(x)
#x =1?
#print(x)
#x =1:
#print(x) 
x =1,
print(x)


#Exercise 3.1

import math

def g(t):
    return math.exp(-t) * math.sin(math.pi * t)

# Print the values of g(0) and g(1)
print("g(0) =", g(0))
print("g(1) =", g(1))



#Exercise 4.2

import sys

# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5.0 / 9.0

# Check if a temperature value was provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python program_name.py <temperature_in_fahrenheit>")
    sys.exit(1)

# Read the temperature in Fahrenheit from the command line
try:
    fahrenheit = float(sys.argv[1])
except ValueError:
    print("Please provide a valid number for the temperature.")
    sys.exit(1)

# Convert the temperature to Celsius
celsius = fahrenheit_to_celsius(fahrenheit)

# Print the result
print(f"The temperature in Celsius is: {celsius:.2f}°C")



# Exercise 4.3


# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5.0 / 9.0

# File name containing the temperature data
filename = "temperature.txt"

try:
    with open(filename, 'r') as file:
        # Read the lines from the file
        lines = file.readlines()
        
        # Find the line that contains "Fahrenheit degrees"
        for line in lines:
            if "Fahrenheit degrees" in line:
                # Extract the Fahrenheit value from the line
                fahrenheit_str = line.split(":")[1].strip()
                fahrenheit = float(fahrenheit_str)
                break
        else:
            # If the loop completes without finding the line
            raise ValueError("Fahrenheit degrees not found in the file.")
        
        # Convert the temperature to Celsius
        celsius = fahrenheit_to_celsius(fahrenheit)
        
        # Print the result
        print(f"The temperature in Celsius is: {celsius:.2f}°C")

except FileNotFoundError:
    print(f"File '{filename}' not found.")
except ValueError as e:
    print(f"Error reading temperature: {e}")


# Exercise 4.17

import calendar
import sys

def get_weekday_name(year, month, day):
    # Get the weekday index: 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
    weekday_index = calendar.weekday(year, month, day)
    
    # List of weekday names
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Return the corresponding weekday name
    return weekdays[weekday_index]

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <year> <month> <day>")
        sys.exit(1)

    try:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        day = int(sys.argv[3])
        
        # Validate the date
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12.")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31.")
        if day > calendar.monthrange(year, month)[1]:
            raise ValueError(f"Day must be between 1 and {calendar.monthrange(year, month)[1]} for the given month.")
        
        weekday_name = get_weekday_name(year, month, day)
        print(weekday_name)
    
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


# python day_of_week.py 2024 9 4



# Exercise 5.1

import numpy as np

# Define the range and number of points
x_start, x_end, num_points = -4, 4, 41

# Generate 41 uniformly spaced x coordinates in [-4, 4]
xlist = np.linspace(x_start, x_end, num_points)

# Define the function h(x)
def h(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# Compute h(x) for each x in xlist
hlist = h(xlist)

# Convert to Python lists (optional, if you need them in list format)
xlist = xlist.tolist()
hlist = hlist.tolist()

# Print the results
print("xlist:", xlist)
print("hlist:", hlist)


# Exercise 5.2

import numpy as np

# Define the range and number of points
x_start, x_end, num_points = -4, 4, 41

# Generate 41 uniformly spaced x coordinates in [-4, 4]
x_values = np.linspace(x_start, x_end, num_points)

# Define the function h(x)
def h(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# Create empty lists for x and y values
x = []
y = []

# Compute each element in x and y using a for loop
for x_val in x_values:
    x.append(x_val)
    y.append(h(x_val))

# Print the results
print("x:", x)
print("y:", y)



#Exercise 5.4

import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x_start, x_end, num_points = -4, 4, 1000
x = np.linspace(x_start, x_end, num_points)

# Define the function h(x)
def h(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# Compute h(x) values
y = h(x)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='$h(x) = \\frac{1}{\sqrt{2\pi}} e^{-\\frac{1}{2}x^2}$', color='purple')
plt.title('Plot of $h(x)$')
plt.xlabel('x')
plt.ylabel('$h(x)$')
plt.legend()
plt.grid(True)
plt.show()


# Excercise 5.5

import numpy as np

# Define the vector
v = np.array([2, 3, -1])

# Define the function f(x)
def f(x):
    return x**3 + x * np.exp(x) + 1

# Apply f to each element of v (element-wise operation)
f_v_manual = np.array([f(x) for x in v])

# Calculate using NumPy vector computing
f_v_numpy = v**3 + v * np.exp(v) + 1

# Output both results
print("f(v) using manual function application:", f_v_manual)
print("f(v) using NumPy vectorized operation:", f_v_numpy)

# Check if both results are equal
print("Are both results equal?", np.allclose(f_v_manual, f_v_numpy))



# Exercise 5.6

x_list = np.array([0,2])
t_list = np.array([1, 1.5])

def y(x, t):
    return np.cos(np.sin(x)) + np.exp(1/t)

y_list = np.array([y(x, t) for x, t in zip(x_list, t_list) ])

print("\n\nThis is x: ", x_list)
print("This is t: ", t_list)
print("This is y: ", y_list)




# Exercise 5.7

w_array = np.array([np.linspace(0, 3, 10)])

print("\n\n\n This is w array: ", w_array)
print("\nThis is w[:]", w_array[:])
print("\nThis is w[:-2]", w_array[:-2]) # not working
print("\nThis is w[::5]", w_array[::5])
print("\nThis is w[2:-2:6]", w_array[2:-2:6]) #not working



# Exercise 5.9

v_0 = 10
g = 9.81
t = np.array([0, 2*v_0/g])

def y_t(t):
    return v_0*t - 1/2*g*t**2

y = v_0*t - 1/2*g*t**2
y_t_list = [y_t(x) for x in t]

print("\n\n\n This is y(t): ", y)
print("This is y_t(t): ", y_t_list)
print("This is t: ", t)



# Exercise 5.11

import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(-10, 10, 100)
y = np.sin(x)

# Create a plot
plt.plot(x, y)

# Set the limits for x and y axes
plt.xlim(-15, 15)  # Set x-axis limits from -15 to 15
plt.ylim(-2, 2)    # Set y-axis limits from -2 to 2

# Add labels and title for clarity
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Example Plot with Specified Axis Limits')

# Display the plot
plt.show()



# Exercise 5.12

f_list = np.array([np.linspace(-20, 120, 5)])

def cel_1(f):
    return (f-30)/2

def cel_2(f):
    return (f-32)*5/9

cel1_list = [cel_1(f) for f in f_list]
cel2_list = [cel_2(f) for f in f_list]

print("\n\n\nThis is first  list: ", cel1_list)
print("\nThis is second list: ", cel2_list)
print("\nThese are f values: ", f_list)




# Exercise 5.14

x_list = [-1, -0.9933, -0.9867, -0.98, -0.9733]
y_list = [-0, -0.0087, -0.0179, -0.0274, -0.0374]

print(f"\n\n\n {'These are x and y columns':>5}")
print(f"\n{'x':>10} {'y':>10}")
print("-"*25)

for x, y in zip(x_list, y_list):
    print(f"{x:>10.4f} {y:>10.4f}")
    
#plt.figure(figsize=(8,10))
#plt.scatter(x_list,y_list)

mean_y= np.mean(y_list)
max_y = np.max(y_list)
min_y = np.min(y_list)

print("\n This is the mean value in y list: ", mean_y)
print("\n This is the maximum value in y list: ", max_y)
print("\n This is the minimum value in y list: ", min_y)

# Plotting the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x_list, y_list, color='blue', label='Data points')

# Highlighting the mean, max, and min on the plot
plt.axhline(y=mean_y, color='green', linestyle='--', label=f'Mean (y = {mean_y:.4f})')
plt.scatter(x_list[y_list.index(max_y)], max_y, color='red', label=f'Maximum (y = {max_y:.4f})')
plt.scatter(x_list[y_list.index(min_y)], min_y, color='orange', label=f'Minimum (y = {min_y:.4f})')

# Annotating the max and min points
plt.text(x_list[y_list.index(max_y)], max_y, f'Max: {max_y:.4f}', fontsize=9, verticalalignment='bottom', color='red')
plt.text(x_list[y_list.index(min_y)], min_y, f'Min: {min_y:.4f}', fontsize=9, verticalalignment='top', color='orange')

# Adding titles and labels
plt.title("Scatter Plot of x_list vs y_list with Mean, Max, and Min Highlights", fontsize=14)
plt.xlabel("x values")
plt.ylabel("y values")

# Displaying the legend
plt.legend()

# Showing the plot
plt.show()



# Exercise 5.28

import numpy as np
import matplotlib.pyplot as plt

# Define the range for x and the initial time t
x = np.linspace(-4, 4, 100)  # Increased the number of points for a smoother curve
t = 0

# Define the function
def wave_packet(x, t):
    return np.exp(-(x - 3 * t)**2) * np.sin(3 * np.pi * (x - t))

# Calculate the function values
y = wave_packet(x, t)

# Plotting
plt.figure(figsize=(8, 6))

plt.scatter(x, y, color="blue", label="Wave Packet")

# Set axis limits
plt.xlim(x[0], x[-1])
plt.ylim(np.min(y), np.max(y))

# Add labels and title
plt.title("Wave Packet", fontsize=16)
plt.xlabel("x values")
plt.ylabel("y values")

# Add legend
plt.legend()

# Show the plot
plt.show()


# Exercise 5.29

import numpy as np
x = np.linspace(0, 2, 20)
y = x*(2- x)
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.title("Wave Packet", fontsize=16)
plt.xlabel("x values")
plt.ylabel("y values")
plt.show()


# Exercise 5.30

import numpy as np
import matplotlib.pyplot as plt

# Define the range for x and the initial time t
T = np.linspace(0, 100, 200)  # Increased the number of points for a smoother curve
A = 2.414 * 10**-5
B = 247.8
C= 140

# Define the function
def viscosity(T, A, B, C):
    return A * 10**(B/(T-C))

# Calculate the function values
y = viscosity(T, A, B, C)
x= T

# Plotting
plt.figure(figsize=(8, 6))

plt.scatter(x, y, color="blue", label="Wave Packet")

# Set axis limits
plt.xlim(x[0], x[-1])
plt.ylim(np.min(y), np.max(y))

# Add labels and title
plt.title("Viscosity of Water", fontsize=16)
plt.xlabel("x values")
plt.ylabel("y values")

# Add legend
plt.legend()

# Show the plot
plt.show()


# Exercise 5.43

import sys
import numpy as np
import matplotlib.pyplot as plt

# Read command-line arguments
func_str = sys.argv[1]
xmin = float(eval(sys.argv[2]))
xmax = float(eval(sys.argv[3]))

# Create x values
x = np.linspace(xmin, xmax, 400)

# Evaluate the function
f = eval('lambda x: ' + func_str)

# Plot the function
plt.figure()
plt.plot(x, f(x))
plt.title(f'Plot of {func_str}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Save the plot to a file
plt.savefig('tmp.png')

# Optionally show the plot
plt.show()


# Exercise 5.44

import sys
import numpy as np
import matplotlib.pyplot as plt

# Default number of points
default_num_points = 501

# Input validation
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("Usage: python program.py <function> <xmin> <xmax> [num_points]")
    sys.exit(1)

func_str = sys.argv[1]

try:
    xmin = float(sys.argv[2])
    xmax = float(sys.argv[3])
    if xmin >= xmax:
        raise ValueError("xmin must be less than xmax.")
except ValueError as e:
    print(f"Invalid input for xmin or xmax: {e}")
    sys.exit(1)

# Optional fourth argument for number of points
if len(sys.argv) == 5:
    try:
        num_points = int(sys.argv[4])
        if num_points <= 0:
            raise ValueError("num_points must be a positive integer.")
    except ValueError as e:
        print(f"Invalid number of points: {e}")
        sys.exit(1)
else:
    num_points = default_num_points  # Set to default if not provided

# Create x values
x = np.linspace(xmin, xmax, num_points)

# Try to evaluate the function
try:
    f = eval('lambda x: ' + func_str)
    # Test if the function works by evaluating it on x[0]
    f(x[0])
except Exception as e:
    print(f"Error evaluating the function: {e}")
    sys.exit(1)

# Plot the function
plt.figure()
plt.plot(x, f(x))
plt.title(f'Plot of {func_str}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Save the plot to a file
plt.savefig('tmp.png')

# Optionally show the plot
plt.show()



# Exercise 5.45

import numpy as np
import matplotlib.pyplot as plt
import sys

# Function to compute the vertical position of the ball at time t
def y(t, v0, g):
    return v0 * t - 0.5 * g * t**2

# Function to compute the velocity (derivative of y with respect to t)
def v(t, v0, g):
    return v0 - g * t

# Constants
g = 9.81  # acceleration due to gravity in m/s^2

# Get input for mass and initial velocity from the command line
if len(sys.argv) != 3:
    print("Usage: python ball_energy.py <mass> <initial_velocity>")
    sys.exit(1)

m = float(sys.argv[1])  # mass in kg
v0 = float(sys.argv[2])  # initial velocity in m/s

# Define the time interval [0, 2v0/g]
t_max = 2 * v0 / g
t = np.linspace(0, t_max, 1000)

# Compute potential energy P(t), kinetic energy K(t), and total energy
P = m * g * y(t, v0, g)  # Potential energy: P = mgy
K = 0.5 * m * v(t, v0, g)**2  # Kinetic energy: K = 1/2 * mv^2
E_total = P + K  # Total energy: P + K

# Plotting P(t), K(t), and P + K(t)
plt.figure(figsize=(10, 6))
plt.plot(t, P, label="Potential Energy (P)", color="blue")
plt.plot(t, K, label="Kinetic Energy (K)", color="red")
plt.plot(t, E_total, label="Total Energy (P + K)", color="green", linestyle='--')

# Adding labels and title
plt.title(f"Energy vs Time for m = {m} kg, v0 = {v0} m/s")
plt.xlabel("Time (t) [s]")
plt.ylabel("Energy [J]")
plt.legend()
plt.grid(True)

# Show plot
plt.show()

"""
# Exercise 5.46

import numpy as np
import matplotlib.pyplot as plt

# Define the function that resembles the "w" shape
def w_function(x):
    return np.sin(2 * x) * np.exp(-0.1 * x**2)

# Test function to verify the output of the w_function
def test_w_function():
    # Known values for testing at specific points
    assert np.isclose(w_function(0), 0), "Test failed at x=0"
    assert np.isclose(w_function(np.pi / 2), 0, atol=1e-1), "Test failed near first peak"
    assert np.isclose(w_function(-np.pi / 2), 0, atol=1e-1), "Test failed near first trough"
    assert np.isclose(w_function(2*np.pi), 0, atol=1e-1), "Test failed near the second peak"
    print("All tests passed.")

# Generate x values for plotting
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

# Plot the function
plt.figure(figsize=(8, 5))
plt.plot(x_vals, w_function(x_vals), label=r'$f(x) = \sin(2x) e^{-0.1x^2}$', color='blue')
plt.title('Function resembling a "w" shape')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.legend()
plt.show()

# Run the test
test_w_function()


# Exercise 5.52

import numpy as np

# Define the 2D array using NumPy
A = np.array([
    [0, 2, -1],
    [-1, -1, 0],
    [0, 5, 0]
])


def f(x):
    return x**3 + x * np.exp(x) + 1

print("\n\nThe result for the calculations is: \n\n")
print(type(A))
print(A**3 + A*np.exp(A) +1)
print("\n\n")


## Exercise 5.53

x=np.linspace(0,1,3)

print(type(x))

















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
plt.text(x_list[y_list.index(max_y)], max_y, f'Max: {max_y:.4f}', 
         fontsize=9, verticalalignment='bottom', color='red')
plt.text(x_list[y_list.index(min_y)], min_y, f'Min: {min_y:.4f}',
         fontsize=9, verticalalignment='top', color='orange')

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





# Exercise 6.1

import requests

def load_constants_from_url(url):
    constants = {}

    # Fetch the file content from the URL
    response = requests.get(url)
    lines = response.text.splitlines()

    # Iterate over each line in the file (skip the header lines)
    for line in lines[2:]:  # Skip the first two lines (header)
        # Strip any extra whitespace from the line
        line = line.strip()
        
        if not line:  # Skip empty lines
            continue

        # Split the line into three parts: name, value, and dimension
        name = line[:25].strip()  # Constant name (first 25 characters)
        value = line[25:45].strip()  # Constant value (next section)
        dimension = line[45:].strip()  # Dimension (rest of the line)

        # Convert the value to a float
        value = float(value)

        # Store the constant in the dictionary
        constants[name] = {
            'value': value,
            'dimension': dimension
        }

    return constants

# Example usage
url = 'https://raw.githubusercontent.com/hplgit/scipro-primer/master/src/dictstring/constants.txt'
constants_dict = load_constants_from_url(url)

# Example: access the value and dimension of the 'gravitational constant'
#print(constants_dict['gravitational constant'])

for key, value in constants_dict.items():
    print(f"{key}: {value}")


# Exercise 6.2

t2 =[]
t2.append(-5)
t2.append(10.5)

print("\n\n")
print(t2)
print("\n\n")
t1 ={}
t1[0]=-5
t1[1] = 10.5
print(t1)

t3 =[]
t3[0]=-5
t3[1] = 10.5


print("\n\n")
#print(t3)



# Exercise 6.5

import requests

def load_starlist_from_url(url):
    stars_dict = {}

    # Fetch the raw file content from the URL
    response = requests.get(url)
    lines = response.text.splitlines()

    # Iterate over each line in the file (skip the header line)
    for line in lines[1:-1]:
        line = line.strip()
        
        if not line:  # Skip empty lines or lines starting with '('
            continue

        # Split the line into parts: star name, distance, apparent magnitude, and luminosity
        parts = line.split(',')  # Split based on the comma delimiter
        print(parts)
       

        if len(parts) == 5:
            # Extract the name (removing any leading/trailing characters like parentheses)
            name = parts[0].strip().strip("'")[2:]
            # Extract the luminosity (fourth element), removing any parentheses and converting to float
            luminosity = float(parts[3].strip().strip(')'))

            # Store the name and luminosity in the dictionary
            stars_dict[name] = luminosity

    return stars_dict

# Correct raw URL for the stars data on GitHub
url = 'https://raw.githubusercontent.com/hplgit/scipro-primer/master/src/funcif/stars.txt'
stars_dict = load_starlist_from_url(url)

# Print the resulting dictionary
for star, luminosity in stars_dict.items():
    print(f"{star}: {luminosity}")



# Exercise 6.6

def create_stars_nested_dict(data):
    stars_luminosity = {}

    for star in data:
        name = star[0]         # Name of the star
        distance = star[1]
        apparaent_brightness = star[2]
        luminosity = star[3]   # Luminosity of the star (third value)
        
        
        stars_luminosity[name] = {
            'distance' : distance,
            'apparent_brightness': apparaent_brightness,
            'luminosity':luminosity
            }
    
    return stars_luminosity

# The provided star data
data = [
    ('Alpha Centauri A',    4.3,  0.26,      1.56),
    ('Alpha Centauri B',    4.3,  0.077,     0.45),
    ('Alpha Centauri C',    4.2,  0.00001,   0.00006),
    ("Barnard's Star",      6.0,  0.00004,   0.0005),
    ('Wolf 359',            7.7,  0.000001,  0.00002),
    ('BD +36 degrees 2147', 8.2,  0.0003,    0.006),
    ('Luyten 726-8 A',      8.4,  0.000003,  0.00006),
    ('Luyten 726-8 B',      8.4,  0.000002,  0.00004),
    ('Sirius A',            8.6,  1.00,      23.6),
    ('Sirius B',            8.6,  0.001,     0.003),
    ('Ross 154',            9.4,  0.00002,   0.0005),
]

# Creating the dictionary
stars_luminosity_dict = create_stars_nested_dict(data)

# Printing the result
print(stars_luminosity_dict['Alpha Centauri A']['luminosity'])


# Exercise 6.7

import requests

def load_human_evolution_from_url(url):
    humans = {}

    # Fetch the raw file content from the URL
    response = requests.get(url)
    lines = response.text.splitlines()
    
    # Skip header lines and iterate over remaining lines
    for line in lines[3:10]:  # Assuming the first two lines are headers
        line = line.strip()
        
        if not line:  # Skip empty lines
            continue

        # Split the line into parts based on multiple spaces
        parts = line.split()
        
        if len(parts) < 5:
            # Handle the case where the line does not have enough parts
            continue
        for i in range(len(parts)):
            parts[i].strip()
        species = parts[0] + " " + parts[1]  # Combine the first two parts as species name
        height = parts[2]
        weight = parts[3]
        brain_volume = parts[4]
        when = " ".join(parts[5:])  # Join remaining parts as the 'when' info
        
        # Store the data in the nested dictionary
        humans[species] = {
            'height': height,
            'weight': weight,
            'brain volume': brain_volume,
            'when': when
        }

    return humans

# Correct raw URL for the human evolution data on GitHub
url = 'https://raw.githubusercontent.com/hplgit/scipro-primer/master/src/dictstring/human_evolution.txt'
humans_dict = load_human_evolution_from_url(url)

# Print the resulting dictionary in a tabular form
print(f"{'Species':<30} {'Height':<10} {'Weight':<10} {'Brain Volume':<15} {'When':<40}")
print("-" * 105)
for species, data in humans_dict.items():
    print(f"{species:<30} {data['height']:<10} {data['weight']:<10} {data['brain volume']:<15} {data['when']:<40}")
    
    
   
# Exercise 6.8

    
        # FILE UNAVILABLE ON GITHUB



#Exercise 6.9

def dict_area_triangle(dict):
    
 A = 1/2*(dict[2][0]*dict[3][1]-dict[3][0]*dict[2][1]
          -dict[1][0]*dict[3][1]+dict[3][0]*dict[1][1]
          +dict[1][0]*dict[2][1]-dict[2][0]*dict[1][1]) 
 return A


dict_1 = {1: (0,1), 2: (1,0), 3: (0,2)}

print(dict_area_triangle(dict_1))



# Exercise 6.10

# Representing the polynomial as a list and dictionary
# Polynomial: -1/2 + 2*x^100

# List representation: each element is a tuple (coefficient, exponent)
polynomial_list = [(-1/2, 0), (2, 100)]

# Dictionary representation: keys are exponents, values are coefficients
polynomial_dict = {0: -1/2, 100: 2}

# Print the representations
print("Polynomial as a list:", polynomial_list)
print("Polynomial as a dictionary:", polynomial_dict)

# Function to evaluate the polynomial using list representation
def evaluate_polynomial_list(poly_list, x):
    result = 0
    for coeff, exp in poly_list:
        result += coeff * (x ** exp)
    return result

# Function to evaluate the polynomial using dictionary representation
def evaluate_polynomial_dict(poly_dict, x):
    result = 0
    for exp, coeff in poly_dict.items():
        result += coeff * (x ** exp)
    return result

# Evaluate the polynomial for x = 1.05
x_value = 1.05
result_from_list = evaluate_polynomial_list(polynomial_list, x_value)
result_from_dict = evaluate_polynomial_dict(polynomial_dict, x_value)

# Print the results
print(f"Evaluation using list at x = {x_value}: {result_from_list}")
print(f"Evaluation using dictionary at x = {x_value}: {result_from_dict}")


# Exercise 6.17

def get_base_counts(dna):
 counts ={'A':0,'T':0, 'G':0, 'C':0}
 x= 0
 
 for base in counts:
  for alp in dna:
      if base == alp:
          x += 1
 return x



print(get_base_counts('SAARCI'))





#Exercise 8.1

import random

# Function to simulate coin flips
def flip_coin(N):
    heads_count = 0  # To count the number of heads
    
    for i in range(N):
        r = random.randint(0, 1)  # Generate 0 (head) or 1 (tail)
        if r == 0:
            print("Head")
            heads_count += 1
        else:
            print("Tail")
    
    print(f"\nTotal number of heads: {heads_count}")

# Simulate flipping a coin N times
N = int(input("Enter the number of times to flip the coin: "))
flip_coin(N)




# Exercise 8.2

import random

# Function to estimate the probability of drawing a number in [0.5, 0.6]
def estimate_probability(N):
    count = 0  # To count how many numbers fall in [0.5, 0.6]
    
    for _ in range(N):
        r = random.random()  # Draw a random number from the interval (0, 1)
        if 0.5 <= r <= 0.6:
            count += 1
    
    # The estimated probability is the ratio of numbers in [0.5, 0.6] to the total number of draws
    probability = count / N
    return probability

# Values of N to test (10^1, 10^2, 10^3, 10^6)
N_values = [10**1, 10**2, 10**3, 10**6]

# Run the experiment for each value of N and print the result
for N in N_values:
    probability = estimate_probability(N)
    print(f"Estimated probability for N = {N}: {probability}")



# Exercise 8.3

import random


def random_color():
    
    colors = ['red', 'green', 'blue', 'pink', 'white', 'black', 'puple', 'orange']
    
    choice = random.choice(colors)
    print(f'The chosen color is: {choice}.')
    
    
random_color()


# Exercise 8.4


def color_prop():
  colors = ['red', 'blue', 'puple', 'yellow']
  r = 0; b = 0
  for i in range(10):
      choice = random.choice(colors)
      if choice == "red":
          r += 1 
      if choice == "blue":
          b += 1 
  if b >= 2:
       b = 2 
  if r >= 2:
      r = 2
  
  return [r, b]


list = color_prop()
r_prob= int(list[0])/10 
b_prob = int(list[1])/10

#print(r_prop)
#print(b_prop)
      
 
print(f"\n\nThe probablity of 2 red balls is {r_prob:.2f} and the probability of 2 blue balls is {b_prob:.2f} .")
      

# Exercise 8.5

def prob_6_1time():
        die = [1, 2 , 3, 4 , 5 , 6]
        
        prob = 0 
        for i in die:
            if i == 6:
                prob += 1 
        return prob/len(die)
    

def prob_6_4time():
     die = [1, 2 , 3, 4 , 5 , 6]
     
     prob = prob_6_1time()
     
     for n in range(4):
             if 6 in die:
                 prob *= 1/len(die) 
     return prob

 
    
print(f"The probability of getting 6 1 time is: {prob_6_1time():.3f}")
print(f"The probability of getting 6 4 times is: {prob_6_4time():.6f}")
         
    
import random

# Function to simulate rolling a die once and checking if it gets a 6
def roll_die():
    return random.randint(1, 6)

# Case 1: Probability of getting a 6 on a single throw
def simulate_single_throw(n_trials):
    count_six = 0
    for _ in range(n_trials):
        if roll_die() == 6:
            count_six += 1
    return count_six / n_trials

# Case 2: Probability of getting 6 four times in a row
def simulate_four_throws(n_trials):
    count_all_sixes = 0
    for _ in range(n_trials):
        if all(roll_die() == 6 for _ in range(4)):  # Check if all 4 rolls are 6
            count_all_sixes += 1
    return count_all_sixes / n_trials

# Case 3: Probability of getting a 6 in the fourth throw after getting three 6s
def simulate_fourth_throw(n_trials):
    count_fourth_six = 0
    for _ in range(n_trials):
        # Simulate the first three throws (all 6s)
        if all(roll_die() == 6 for _ in range(3)):
            # Check the fourth throw
            if roll_die() == 6:
                count_fourth_six += 1
    return count_fourth_six / n_trials

# Simulation parameters
n_trials = 100000  # Increase for more accuracy

# Run simulations
prob_case_1 = simulate_single_throw(n_trials)
prob_case_2 = simulate_four_throws(n_trials)
prob_case_3 = simulate_fourth_throw(n_trials)

# Output the probabilities
print(f"Simulated probability of getting a 6 on a single throw: {prob_case_1}")
print(f"Simulated probability of getting 6 four times in a row: {prob_case_2}")
print(f"Simulated probability of getting a 6 on the fourth throw after three 6s: {prob_case_3}")


  
        
# Exercise 8.6

def n_dice(n):
      die = [1, 2 , 3, 4 , 5 , 6]
      
      prob_6 = 1 if 6 in die else 0
      prob = 1
      
      
      for i in range(n):
              prob *= prob_6/len(die)
      return prob
  
n = int(input("Insert a number for turns: "))
print(f"The probability is: {n_dice(n):.8f}")
           
    

# Exercise 8.8

import random 

def fair_dicer():
 r = 10
 die = [1, 2 , 3, 4 , 5 , 6] 
 dice = 0 
 
 for i in range(4):
     dice += random.choice(die)
     print("\n", dice)
 
    
 if dice < 9:
     print(f"\nYou are getting paid {r} euros.")
     
 else:
     print("\nYou need to pay 1 euro")
     

fair_dicer()
     

# Exercise 8.14

import numpy as np
import random

def amuse_park(n):
    # Check for valid input first
    while not (4 <= n <= 10):
        n = int(input("\nPlease enter a value for n between 4 and 10: "))

    # Initialize variables
    balls = ["red", "yellow", "green", "brown"]
    red = 0
    yellow = 0
    green = 0
    brown = 0

    # Run the simulation n times
    for i in range(n):
        choice = random.choice(balls)
        if choice == "red":
            red += 1
            if red == 5:
                balls.remove("red")
        elif choice == "yellow":
            yellow += 1
            if yellow == 5:
                balls.remove("yellow")
        elif choice == "brown":
            brown += 1
            if brown == 7:
                balls.remove("brown")
        elif choice == "green":
            green += 1
            if green == 3:
                balls.remove("green")

    # Check the conditions and print the results
    if red == 3:
        print("\nYou won 60 euros!!")
    if brown >= 3:
        print(f"\nYou won {7 + 5 * np.sqrt(n)} euros!!")
    if yellow == 1 and brown == 1:  # Corrected the logical AND
        print(f"\nYou won {n**3 - 26} euros!!")
    if red >= 1 and yellow >= 1 and brown >= 1 and green >= 1:
        print("\nYou won 23 euros!!")
        
    print("\n\n Done!")

# Get user input and call the function
n = int(input("\nPlease enter a value for n between 4 and 10: "))
amuse_park(n)



# Exercise 8.15

import random

def two_dice_random(n):
  die = [1, 2 , 3, 4 , 5 , 6]
  result = []
  count = 0
  s= []  
  for x in range(2, 13):
      s.append(x)
  
  for i in range(n):
      choice1 = random.choice(die)
      choice2 = random.choice(die)
      sum = choice1 + choice2
      result.append(sum)
      if sum in s:
          count += 1 
  print(f"\nThe sum was in the interval {count} times! \n")
  print(result)
  

n = int(input("\nPlease enter a number for the throwing of the dices: "))
two_dice_random(n)


## CHAT GPT 8.15

import random
import numpy as np

# Initialize variables
num_trials = 100000  # Large number of trials
sum_counts = {i: 0 for i in range(2, 13)}  # To store the counts of each sum

# Simulate the dice rolls
for _ in range(num_trials):
    die1 = random.randint(1, 6)
    die2 = random.randint(1, 6)
    dice_sum = die1 + die2
    sum_counts[dice_sum] += 1

# Calculate experimental probabilities
experimental_probabilities = {k: v / num_trials for k, v in sum_counts.items()}

# Theoretical probabilities (computed from possible outcomes)
theoretical_probabilities = {
    2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36, 7: 6/36,
    8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
}

# Print the results
print(f"{'Sum':<5} {'Experimental Probability':<25} {'Theoretical Probability':<25}")
for s in range(2, 13):
    print(f"{s:<5} {experimental_probabilities[s]:<25} {theoretical_probabilities[s]:<25}")
      

# Exercise 8.16

import numpy as np

N = 1000  # Number of coin flips
r = np.random.rand(N)  # Generate N random numbers between 0 and 1
tails = np.sum(r <= 0.5)  # Count how many are <= 0.5 (representing tails)

print(f"Number of tails: {tails}")



# Exercise 8.17

import numpy as np

# Define the values of N to test
Ns = [10**i for i in [1, 2, 3, 6]]

for N in Ns:
    # Generate N random numbers between 0 and 1
    r = np.random.rand(N)
    
    # Count how many numbers fall in the interval [0.5, 0.6]
    M = np.sum(np.logical_and(r >= 0.5, r <= 0.6))
    
    # Compute the probability
    probability = M / N
    
    # Print the results
    print(f"For N = {N}:")
    print(f"Number of values between 0.5 and 0.6: {M}")
    print(f"Probability: {probability}\n")
    
    

# Exercise 8.27

import random

def rand_guess(n):
# Generate a random integer between x and y (inclusive)
    r = random.randint(1, 100)
    
    while n != r:    
        if n < r:
            r = random.randint(n+1, 100)
            n = int(input("\nIncorrect! Try again with a larger number: "))
        if n > r:
            r = random.randint(1, n-1)
            n = int(input("\nIncorrect! Try again with a smaller number: "))
            
    if n == r:
             print("\nHurray!! You guessed right!!")
             
             
n = int(input("\nInsert a guess please between 1 and 100: "))
rand_guess(n)
        


# CHAT-GPT Excericise 8.27 

import random

# Define the game where the computer guesses the secret number
def play_game(strategy, secret_number):
    p = 1  # lower bound
    q = 100  # upper bound
    guesses = 0

    while True:
        guesses += 1
        
        # Use the chosen strategy to make a guess
        if strategy == 'midpoint':
            guess = (p + q) // 2
        elif strategy == 'random':
            guess = random.randint(p, q)
        
        # Check if the guess is correct
        if guess == secret_number:
            return guesses
        
        # Update the interval [p, q] based on feedback
        if guess < secret_number:
            p = guess + 1
        else:
            q = guess - 1

# Function to simulate the game for both strategies
def simulate_games(num_games):
    total_guesses_midpoint = 0
    total_guesses_random = 0
    
    for _ in range(num_games):
        # Generate a secret number for each game
        secret_number = random.randint(1, 100)
        
        # Play the game with the midpoint strategy
        guesses_midpoint = play_game('midpoint', secret_number)
        total_guesses_midpoint += guesses_midpoint
        
        # Play the game with the random strategy
        guesses_random = play_game('random', secret_number)
        total_guesses_random += guesses_random
    
    # Compute the average number of guesses for each strategy
    avg_guesses_midpoint = total_guesses_midpoint / num_games
    avg_guesses_random = total_guesses_random / num_games
    
    print(f"Average guesses with midpoint strategy: {avg_guesses_midpoint:.2f}")
    print(f"Average guesses with random strategy: {avg_guesses_random:.2f}")

# Run the simulation for a large number of games
simulate_games(10000)



# Exercise 8.28

import numpy as np
import sys

def simulate_game(r, N):
    # Simulate rolling 4 dice N times. Each die roll is between 1 and 6.
    dice_rolls = np.random.randint(1, 7, (N, 4))
    
    # Calculate the sum of the dice rolls for each game
    dice_sums = np.sum(dice_rolls, axis=1)
    
    # Check if the sum is less than 9
    wins = dice_sums < 9
    
    # Calculate the total profit or loss
    # If you win, you gain r euros (but lose the 1 euro investment, so net is r - 1).
    # If you lose, you lose the 1 euro.
    profit = np.where(wins, r - 1, -1)
    
    # Sum up all the profits/losses
    total_profit = np.sum(profit)
    
    # Average profit per game
    avg_profit = total_profit / N
    
    return total_profit, avg_profit

# Read r and N from command line arguments
if __name__ == "__main__":
    # Set default values for r and N in case not provided
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    
    total_profit, avg_profit = simulate_game(r, N)
    
    print(f"Total profit after {N} games: {total_profit} euros")
    print(f"Average profit per game: {avg_profit:.4f} euros")

    if avg_profit > 0:
        print("In the long run, you will win money by playing this game.")
    else:
        print("In the long run, you will lose money by playing this game.")

"""


# Exercise 7.1
import math

class F:
    def __init__(self, a, w):
        """
        Initialize the class with parameters a and w.
        """
        self.a = a
        self.w = w
    
    def calculate(self, x):
        """
        Calculate f(x) = e^(-ax) * sin(wx)
        """
        # Exponential component
        exp_component = math.exp(-self.a * x)
        
        # Sine component
        sin_component = math.sin(self.w * x)
        
        # Calculate the function value
        return exp_component * sin_component

# Example usage:
# Create an instance of the function with a = 2, w = 3
f = F(2, 3)

# Compute the function for a given value of x
result = f.calculate(1)  # for x = 1
print(result)


#Exercise 7.4
import math

class Rectangle:
    def __init__(self, x0, y0, W, H):
        """
        Initialize the rectangle with lower-left corner (x0, y0), width W, and height H.
        """
        self.x0 = x0
        self.y0 = y0
        self.W = W
        self.H = H
    
    def area(self):
        """
        Calculate the area of the rectangle.
        Area = width * height
        """
        return self.W * self.H
    
    def perimeter(self):
        """
        Calculate the perimeter of the rectangle.
        Perimeter = 2 * (width + height)
        """
        return 2 * (self.W + self.H)

class Triangle:
    def __init__(self, x0, y0, x1, y1, x2, y2):
        """
        Initialize the triangle with vertices (x0, y0), (x1, y1), and (x2, y2).
        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def area(self):
        """
        Calculate the area of the triangle using the formula:
        A = 1/2 * |x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1)|
        """
        return 0.5 * abs(self.x0*(self.y1 - self.y2) + self.x1*(self.y2 - self.y0) + self.x2*(self.y0 - self.y1))
    
    def perimeter(self):
        """
        Calculate the perimeter of the triangle by summing the lengths of its sides.
        """
        side1 = math.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2)
        side2 = math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
        side3 = math.sqrt((self.x2 - self.x0)**2 + (self.y2 - self.y0)**2)
        return side1 + side2 + side3

# Test functions
def test_Rectangle():
    """
    Test the Rectangle class by checking area and perimeter.
    """
    rect = Rectangle(0, 0, 4, 5)  # width=4, height=5
    expected_area = 20  # 4 * 5
    expected_perimeter = 18  # 2 * (4 + 5)
    assert math.isclose(rect.area(), expected_area, rel_tol=1e-9), f"Expected area {expected_area}, got {rect.area()}"
    assert math.isclose(rect.perimeter(), expected_perimeter, rel_tol=1e-9), f"Expected perimeter {expected_perimeter}, got {rect.perimeter()}"
    print("Rectangle tests passed!")

def test_Triangle():
    """
    Test the Triangle class by checking area and perimeter.
    """
    tri = Triangle(0, 0, 4, 0, 0, 3)  # right-angled triangle with vertices (0,0), (4,0), (0,3)
    expected_area = 6  # 1/2 * base * height = 1/2 * 4 * 3
    expected_perimeter = 12  # 3 + 4 + 5 (Pythagoras: hypotenuse = 5)
    assert math.isclose(tri.area(), expected_area, rel_tol=1e-9), f"Expected area {expected_area}, got {tri.area()}"
    assert math.isclose(tri.perimeter(), expected_perimeter, rel_tol=1e-9), f"Expected perimeter {expected_perimeter}, got {tri.perimeter()}"
    print("Triangle tests passed!")

# Running the test functions
test_Rectangle()
test_Triangle()

          

 
#Exercise 7.5
import math

class Quadratic:
    def __init__(self, a, b, c):
        """
        Initialize the quadratic function with coefficients a, b, and c.
        """
        self.a = a
        self.b = b
        self.c = c
    
    def value(self, x):
        """
        Compute the value of the quadratic function f(x) = ax^2 + bx + c at a given x.
        """
        return self.a * x**2 + self.b * x + self.c
    
    def table(self, L, R, nx):
        """
        Print a table of x and f(x) values for nx values in the interval [L, R].
        """
        step = (R - L) / (nx - 1)
        print(f"{'x':>10} {'f(x)':>10}")
        print("-" * 20)
        for i in range(nx):
            x = L + i * step
            print(f"{x:10.4f} {self.value(x):10.4f}")
    
    def roots(self):
        """
        Compute the roots of the quadratic equation ax^2 + bx + c = 0 using the quadratic formula.
        Roots are calculated as:
        x1, x2 = (-b ± sqrt(b^2 - 4ac)) / (2a)
        Returns a tuple with two values (root1, root2).
        """
        discriminant = self.b**2 - 4 * self.a * self.c
        if discriminant < 0:
            raise ValueError("The equation has no real roots.")
        sqrt_disc = math.sqrt(discriminant)
        root1 = (-self.b + sqrt_disc) / (2 * self.a)
        root2 = (-self.b - sqrt_disc) / (2 * self.a)
        return root1, root2

# Test function to verify the value and roots methods
def test_Quadratic():
    """
    Test the value and roots methods of the Quadratic class.
    """
    # Test 1: Check the value at a specific point for a known quadratic function
    q = Quadratic(1, -3, 2)  # f(x) = x^2 - 3x + 2
    x = 1
    expected_value = 0  # f(1) = 1^2 - 3*1 + 2 = 0
    assert math.isclose(q.value(x), expected_value, rel_tol=1e-9), f"Expected {expected_value}, got {q.value(x)}"

    # Test 2: Check the roots for a known quadratic function
    # Roots of x^2 - 3x + 2 are 1 and 2
    expected_roots = (2, 1)  # The roots can be in any order
    roots = q.roots()
    assert math.isclose(roots[0], expected_roots[0], rel_tol=1e-9) or math.isclose(roots[0], expected_roots[1], rel_tol=1e-9), f"Expected root1 {expected_roots[0]}, got {roots[0]}"
    assert math.isclose(roots[1], expected_roots[1], rel_tol=1e-9) or math.isclose(roots[1], expected_roots[0], rel_tol=1e-9), f"Expected root2 {expected_roots[1]}, got {roots[1]}"

    print("All tests passed!")

# To use the class, you can import it from this module:
# from Quadratic import Quadratic
     
     
     
     
# Exercise 7.6
class Line:
    def __init__(self, p1, p2):
        """
        Initialize the line with two points p1 and p2.
        Points are 2-tuples or 2-lists representing (x, y).
        """
        self.x1, self.y1 = p1
        self.x2, self.y2 = p2

        # Calculate the slope (m) and intercept (b)
        self.slope = (self.y2 - self.y1) / (self.x2 - self.x1)
        self.intercept = self.y1 - self.slope * self.x1

    def value(self, x):
        """
        Compute the y value on the line for a given x.
        The line equation is y = mx + b, where m is the slope and b is the intercept.
        """
        return self.slope * x + self.intercept

def test_Line():
    """
    Test the Line class by checking the value method for a known line.
    """
    # Create a line through points (0, -1) and (2, 4)
    line = Line((0, -1), (2, 4))

    # Check the value method at specific x coordinates
    assert abs(line.value(0.5) - 0.25) < 1e-9, f"Expected 0.25, got {line.value(0.5)}"
    assert abs(line.value(0) + 1) < 1e-9, f"Expected -1.0, got {line.value(0)}"
    assert abs(line.value(1) - 1.5) < 1e-9, f"Expected 1.5, got {line.value(1)}"

    print("All tests passed!")

# Demonstration
if __name__ == "__main__":
    # Interactive session
    line = Line((0, -1), (2, 4))
    print(line.value(0.5), line.value(0), line.value(1))  # Expected: 0.25 -1.0 1.5

    # Run the test function
    test_Line()

     
 
     
# Exercise 7.7
class Line:
    def __init__(self, *args):
        """
        Initialize the line using one of the following methods:
        1. Two points (p1, p2): Each point is a 2-tuple or 2-list (x, y).
        2. Point and slope (p1, slope): p1 is a point (x1, y1), slope is a number.
        3. Slope and intercept (slope, intercept): Both are numbers.
        """
        if len(args) == 2:
            if isinstance(args[0], (list, tuple)) and isinstance(args[1], (list, tuple)):
                # Two points provided (p1 and p2)
                p1, p2 = args
                self.x1, self.y1 = p1
                self.x2, self.y2 = p2
                self.slope = (self.y2 - self.y1) / (self.x2 - self.x1)  # Calculate slope
                self.intercept = self.y1 - self.slope * self.x1         # Calculate intercept
            elif isinstance(args[0], (list, tuple)) and isinstance(args[1], (int, float)):
                # Point and slope provided (p1, slope)
                p1, slope = args
                self.x1, self.y1 = p1
                self.slope = slope
                self.intercept = self.y1 - self.slope * self.x1  # Calculate intercept from point
            elif isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
                # Slope and intercept provided (slope, intercept)
                self.slope = args[0]
                self.intercept = args[1]
            else:
                raise ValueError("Invalid input types for the two-argument form.")
        else:
            raise ValueError("You must provide either two points, a point and a slope, or a slope and an intercept.")
    
    def value(self, x):
        """
        Compute the y value on the line for a given x.
        The line equation is y = mx + b, where m is the slope and b is the intercept.
        """
        return self.slope * x + self.intercept

# Test function to verify the flexibility of the Line class
def test_Line():
    """
    Test the Line class by checking initialization and value method for different forms of input.
    """
    # Test 1: Two points
    line1 = Line((0, -1), (2, 4))  # Line through points (0, -1) and (2, 4)
    assert abs(line1.value(0.5) - 0.25) < 1e-9, f"Expected 0.25, got {line1.value(0.5)}"
    assert abs(line1.value(0) + 1) < 1e-9, f"Expected -1.0, got {line1.value(0)}"
    assert abs(line1.value(1) - 1.5) < 1e-9, f"Expected 1.5, got {line1.value(1)}"

    # Test 2: Point and slope
    line2 = Line((1, 1), 2)  # Line through point (1, 1) with slope 2
    assert abs(line2.value(0) - (-1)) < 1e-9, f"Expected -1.0, got {line2.value(0)}"
    assert abs(line2.value(2) - 3) < 1e-9, f"Expected 3.0, got {line2.value(2)}"

    # Test 3: Slope and intercept
    line3 = Line(3, -5)  # Line with slope 3 and intercept -5
    assert abs(line3.value(0) + 5) < 1e-9, f"Expected -5.0, got {line3.value(0)}"
    assert abs(line3.value(1) - (-2)) < 1e-9, f"Expected -2.0, got {line3.value(1)}"

    print("All tests passed!")

# Interactive demo
if __name__ == "__main__":
    # Creating lines using different methods
    line1 = Line((0, -1), (2, 4))  # Two points
    line2 = Line((1, 1), 2)        # Point and slope
    line3 = Line(3, -5)            # Slope and intercept

    print("line1:", line1.value(0.5), line1.value(0), line1.value(1))  # 0.25 -1.0 1.5
    print("line2:", line2.value(0), line2.value(2))                    # -1.0 3.0
    print("line3:", line3.value(0), line3.value(1))                    # -5.0 -2.0

    # Run the test function to check if everything works correctly
    test_Line()



# Exercise 7.10
class Hello:
    def __init__(self):
        self.default_greeting = "World"

    def __call__(self, name=None):
        """
        This method allows the object to be called like a function.
        If a name is passed, it prints "Hello, name!".
        """
        if name:
            return f"Hello, {name}!"
        else:
            return self.__str__()

    def __str__(self):
        """
        This method defines what happens when we print the object.
        In this case, it returns "Hello, World!" by default.
        """
        return f"Hello, {self.default_greeting}!"

# Test the class
if __name__ == "__main__":
    a = Hello()
    print(a('students'))  # Should print "Hello, students!"
    print(a)              # Should print "Hello, World!"



# Exercise 7.11
import math

class F2:
    def __init__(self, a, w):
        """
        Initialize the class with parameters a and w.
        """
        self.a = a
        self.w = w

    def __call__(self, x):
        """
        Makes the class instance callable and calculates f(x) = e^(-ax) * sin(wx).
        """
        return self.calculate(x)

    def calculate(self, x):
        """
        Calculate f(x) = e^(-ax) * sin(wx).
        """
        # Exponential component
        exp_component = math.exp(-self.a * x)
        
        # Sine component
        sin_component = math.sin(self.w * x)
        
        # Calculate the function value
        return exp_component * sin_component

    def __str__(self):
        """
        Return a string representation of the function.
        """
        return "exp(-a*x)*sin(w*x)"



# Exercise 7.14

class Rope:
    def __init__(self, knots):
        """
        Initialize the Rope with a given number of knots.
        """
        self.knots = knots

    def __add__(self, other):
        """
        Define the addition operation for combining two ropes.
        Each rope adds one extra knot at the junction when combined.
        """
        # Add the knots from both ropes plus 1 for the junction
        return Rope(self.knots + other.knots + 1)

    def __str__(self):
        """
        Return the number of knots as a string for easy printing.
        """
        return str(self.knots)

# Test function
def test_rope_addition():
    """
    Test function for verifying the Rope class addition operator.
    """
    rope1 = Rope(2)
    rope2 = Rope(2)
    rope3 = rope1 + rope2
    assert rope3.knots == 5, f"Expected 5 knots, but got {rope3.knots}"
    print("Test passed: Rope addition works correctly.")

# Example usage:
if __name__ == "__main__":
    rope1 = Rope(2)
    rope2 = Rope(2)
    rope3 = rope1 + rope2
    print(rope3)  # Expected output: 5
    test_rope_addition()  # Running the test function





#Exercise 7.15

"""

To implement the += and -= operators in the Account class for deposit and withdraw operations, we will need to:

Implement the __iadd__ method for the += operator (deposit).
Implement the __isub__ method for the -= operator (withdrawal).
Define the __str__ method for a user-friendly string representation of the account.
Define the __repr__ method for a more detailed and developer-friendly representation.
Write a test_Account() function to verify all functionalities.


Why use !r?
In contexts like debugging or logging, 
it's helpful to know exactly what type of data you’re dealing with. 
The repr() function is designed to give detailed, 
often more precise information about objects 
(like showing that self.owner is a string).


"""


class Account:
    def __init__(self, owner, balance=0):
        """
        Initialize the Account with an owner and an optional balance (default 0).
        """
        self.owner = owner
        self.balance = balance

    def __iadd__(self, amount):
        """
        Implement the in-place addition operator for deposit (+=).
        Deposit the specified amount to the account.
        """
        if amount > 0:
            self.balance += amount
        return self  # Return self to make += work

    def __isub__(self, amount):
        """
        Implement the in-place subtraction operator for withdraw (-=).
        Withdraw the specified amount from the account.
        """
        if 0 <= amount <= self.balance:
            self.balance -= amount
        else:
            raise ValueError("Insufficient funds or invalid withdrawal amount.")
        return self  # Return self to make -= work

    def __str__(self):
        """
        Return a user-friendly string representation of the account.
        """
        return f"Account owner: {self.owner}, Balance: {self.balance:.2f}"

    def __repr__(self):
        """
        Return a detailed, developer-friendly string representation of the account.
        """
        return f"Account(owner={self.owner!r}, balance={self.balance:.2f})"

# Test function
def test_Account():
    """
    Test the Account class with the += and -= operators, and ensure everything works.
    """
    # Create an account for "Alice" with a starting balance of 100
    account = Account(owner="Alice", balance=100)
    
    # Test the deposit using +=
    account += 50  # Alice deposits 50
    assert account.balance == 150, f"Expected balance: 150, got: {account.balance}"

    # Test the withdraw using -=
    account -= 30  # Alice withdraws 30
    assert account.balance == 120, f"Expected balance: 120, got: {account.balance}"

    # Test insufficient funds withdrawal (should raise an error)
    try:
        account -= 200  # This should raise an exception
    except ValueError as e:
        print(f"Expected error: {e}")

    # Print account details
    print(account)  # Expected: "Account owner: Alice, Balance: 120.00"
    print(repr(account))  # Expected: "Account(owner='Alice', balance=120.00)"

    print("All tests passed.")

# Example usage:
if __name__ == "__main__":
    test_Account()




# Exercise 7.21
import numpy as np
import matplotlib.pyplot as plt

class PiecewiseConstant:
    def __init__(self, intervals, xmax):
        """
        Initializes the PiecewiseConstant function.
        
        Parameters:
        intervals (list of tuples): A list of (length, value) tuples.
        xmax (float): The maximum x value for the piecewise function.
        """
        self.intervals = intervals
        self.xmax = xmax
        self.x_values = self._create_x_values()
        self.y_values = self._create_y_values()

    def _create_x_values(self):
        """
        Creates the x values for the piecewise function.
        """
        x_values = []
        current_x = 0.0
        
        for length, _ in self.intervals:
            current_x += length
            if current_x <= self.xmax:
                x_values.append(current_x)
        
        # Ensure that we don't exceed xmax
        if self.xmax not in x_values:
            x_values.append(self.xmax)
        
        return x_values

    def _create_y_values(self):
        """
        Creates the y values for the piecewise function.
        """
        y_values = []
        for _, value in self.intervals:
            y_values.append(value)
        
        # Add the last value for the last segment to xmax
        if self.xmax > self.x_values[-1]:
            y_values.append(y_values[-1])  # extend the last value to xmax
        
        return y_values

    def __call__(self, x):
        """
        Evaluates the piecewise constant function at a given x.
        
        Parameters:
        x (float or array-like): The input value(s).
        
        Returns:
        float or np.ndarray: The output value(s).
        """
        if np.isscalar(x):
            return self._evaluate(x)
        else:
            return np.array([self._evaluate(val) for val in x])

    def _evaluate(self, x):
        """
        Helper method to evaluate the function at a single point.
        """
        if x < 0 or x > self.xmax:
            raise ValueError(f"x={x} is out of bounds [0, {self.xmax}]")
        
        # Find the appropriate interval
        cumulative_length = 0.0
        for i, (length, value) in enumerate(self.intervals):
            cumulative_length += length
            if x <= cumulative_length:
                return value
        return self.y_values[-1]  # Return the last value if x is at xmax

    def plot(self):
        """
        Plots the piecewise constant function.
        
        Returns:
        tuple: (x_values, y_values) for plotting.
        """
        x_plot = []
        y_plot = []
        
        current_x = 0.0
        for (length, value) in self.intervals:
            x_plot.append(current_x)
            x_plot.append(current_x + length)
            y_plot.append(value)
            y_plot.append(value)
            current_x += length
        
        # Extend to xmax
        if current_x < self.xmax:
            x_plot.append(self.xmax)
            y_plot.append(y_plot[-1])
        
        return np.array(x_plot), np.array(y_plot)

# Example usage:
f = PiecewiseConstant([(0.4, 1), (0.2, 1.5), (0.1, 3)], xmax=4)
print(f(1.5), f(1.75), f(4))

# For array input
x = np.linspace(0, 4, 21)
print(f(x))

# Plotting the function
x_values, y_values = f.plot()
plt.plot(x_values, y_values, drawstyle='steps-post')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Piecewise Constant Function')
plt.grid()
plt.show()


#  Exercise 7.28
class Polynomial:
    def __init__(self, coeff_dict):
        """
        Initializes the Polynomial with a dictionary of coefficients.
        coeff_dict: A dictionary where the key is the exponent and the value is the coefficient.
        Example: {4: 1, 2: -2, 0: 3} represents the polynomial x^4 - 2x^2 + 3.
        """
        self.coeff = coeff_dict

    def __call__(self, x):
        """
        Evaluates the polynomial at a given value of x.
        """
        result = 0
        for exp, coef in self.coeff.items():
            result += coef * (x ** exp)
        return result
    
    def __add__(self, other):
        """
        Adds two Polynomial objects.
        """
        result = self.coeff.copy()
        
        for exp, coef in other.coeff.items():
            if exp in result:
                result[exp] += coef
            else:
                result[exp] = coef
        
        # Remove terms with 0 coefficient
        result = {exp: coef for exp, coef in result.items() if coef != 0}
        
        return Polynomial(result)
    
    def __sub__(self, other):
        """
        Subtracts two Polynomial objects.
        """
        result = self.coeff.copy()
        
        for exp, coef in other.coeff.items():
            if exp in result:
                result[exp] -= coef
            else:
                result[exp] = -coef
        
        # Remove terms with 0 coefficient
        result = {exp: coef for exp, coef in result.items() if coef != 0}
        
        return Polynomial(result)
    
    def __mul__(self, other):
        """
        Multiplies two Polynomial objects.
        """
        result = {}
        
        for exp1, coef1 in self.coeff.items():
            for exp2, coef2 in other.coeff.items():
                new_exp = exp1 + exp2
                new_coef = coef1 * coef2
                
                if new_exp in result:
                    result[new_exp] += new_coef
                else:
                    result[new_exp] = new_coef
        
        # Remove terms with 0 coefficient
        result = {exp: coef for exp, coef in result.items() if coef != 0}
        
        return Polynomial(result)
    
    

# Test __mul__
p1 = Polynomial({0: 1, 3: 1})  # 1 + x^3
p2 = Polynomial({1: -2, 2: 3}) # -2*x + 3*x^2
p3 = p1 * p2                    # -2*x + 3*x^2 - 2*x^4 + 3*x^5
print(p3.coeff)  # Should print {1: -2, 2: 3, 4: -2, 5: 3}



# Test __sub__
p1 = Polynomial({4: 1, 2: -2, 0: 3})  # x^4 - 2*x^2 + 3
p2 = Polynomial({0: 4, 1: 3})         # 4 + 3*x
p3 = p1 - p2                           # x^4 - 2*x^2 - 3*x - 1
print(p3.coeff)  # Should print {4: 1, 2: -2, 1: -3, 0: -1}


# Test __add__
p1 = Polynomial({4: 1, 2: -2, 0: 3})  # x^4 - 2*x^2 + 3
p2 = Polynomial({0: 4, 1: 3})         # 4 + 3*x
p3 = p1 + p2                           # x^4 - 2*x^2 + 3*x + 7
print(p3.coeff)  # Should print {4: 1, 2: -2, 1: 3, 0: 7}


# Test __call__
p1_dict = {4: 1, 2: -2, 0: 3}  # Polynomial x^4 - 2*x^2 + 3
p1 = Polynomial(p1_dict)
print(p1(2))  # Should print 11 (16 - 8 + 3)



def test_call():
    p1 = Polynomial({4: 1, 2: -2, 0: 3})  # x^4 - 2*x^2 + 3
    assert p1(2) == 11, "Test __call__ failed"
    print("Test __call__ passed")

def test_add():
    p1 = Polynomial({4: 1, 2: -2, 0: 3})  # x^4 - 2*x^2 + 3
    p2 = Polynomial({0: 4, 1: 3})         # 4 + 3*x
    p3 = p1 + p2                           # x^4 - 2*x^2 + 3*x + 7
    assert p3.coeff == {4: 1, 2: -2, 1: 3, 0: 7}, "Test __add__ failed"
    print("Test __add__ passed")

def test_mul():
    p1 = Polynomial({0: 1, 3: 1})         # 1 + x^3
    p2 = Polynomial({1: -2, 2: 3})        # -2*x + 3*x^2
    p3 = p1 * p2                           # -2*x + 3*x^2 - 2*x^4 + 3*x^5
    assert p3.coeff == {1: -2, 2: 3, 4: -2, 5: 3}, "Test __mul__ failed"
    print("Test __mul__ passed")

# Run tests
test_call()
test_add()
test_mul()

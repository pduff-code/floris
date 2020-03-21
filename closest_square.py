

# taken from: 
#  https://www.geeksforgeeks.org/closest-perfect-square-and-its-distance/
# by sahishelangia     
# Python script to find the closest  
# perfect square taking minimum steps 
# to reach from a number  
  
# Function to check if a number is  
# perfect square or not 
from math import sqrt, floor  
def isPerfect(N): 
    if (sqrt(N) - floor(sqrt(N)) != 0): 
        return False
    return True
  
# Function to find the closest perfect square  
# taking minimum steps to reach from a number  
def getClosestPerfectSquare(N): 
    if (isPerfect(N)):  
        print(N, "0")  
        return
  
    # Variables to store first perfect  
    # square number above and below N  
    aboveN = -1
    belowN = -1
    n1 = 0
  
    # Finding first perfect square  
    # number greater than N  
    n1 = N + 1
    while (True): 
        if (isPerfect(n1)): 
            aboveN = n1  
            break
        else: 
            n1 += 1
  
    # Finding first perfect square  
    # number less than N  
    n1 = N - 1
    while (True):  
        if (isPerfect(n1)):  
            belowN = n1  
            break
        else: 
            n1 -= 1
              
    # Variables to store the differences  
    diff1 = aboveN - N  
    diff2 = N - belowN  
  
    if (diff1 > diff2): 
        print(belowN, diff2)  
    else: 
        print(aboveN, diff1) 
  
# Driver code  
N = 1522
getClosestPerfectSquare(N) 
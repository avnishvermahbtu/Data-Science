'''
Real-Word Example: Multiprocessing for CPU-bound Tasks
Scenario: Factotial Calculation
Factorial calculation .especially for large number,
involve significant computational work.Multiprocessing 
can be used to distribute the workload across multiple 
CPU cores ,improving performance.

'''
import multiprocessing
import math
import sys
import time

## Increase the maximum number of digits for integer conversion
sys.set_int_max_str_digits(100000)

## function to compute factorials of a give number

def compute_factorial(number):
    print(f"Computing factorial of {number} ")
    result=math.factorial(number)
    print(f"Computing factorial of {number} is {result}")
    return result
    
    
if __name__=="__main__":
        numbers=[6000,5000,700,8000]
        
        start_time=time.time()
        
        ## Create a pool of worker process
        with multiprocessing.Pool() as pool:
            results=pool.map(compute_factorial,numbers)
            
        end_time=time.time()
        print(f"Result: {results}")
        print(f"Time taken:{end_time-start_time} seconds")   
         




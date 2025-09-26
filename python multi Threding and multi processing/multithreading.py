## Multitreading
# When to use Multi Threading  
## I/O-bound tasks: Tasks that spend more time waiting for I/O operation(e.g, file operation,network request) 
## concurrent execution: When you want to improve the throughput of your application by performing  multiple operation concurrently.

## if two code both code parrall 
 
import threading
import time

def print_numbers():
    for i in range(5):
        time.sleep(2)
        print(f"Number:{i}")   
        
def print_letter():
    for letter in "abcde":
       time.sleep(2)
       print(f"Letter:{letter}")

## Create 2 threads
t1=threading.Thread(target=print_numbers)  ##    (target,args)
t2=threading.Thread(target=print_letter)

t=time.time()
## Start the threads // : Create a Thread
t1.start()
t2.start()
   
## Wait for the threads to  complete //End the thread Execution
t1.join()
t2.join()   
finished_time=time.time()-t
print(finished_time)    
       
       
    
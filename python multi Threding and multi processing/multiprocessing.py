## Processes that run in Parallel
## CPU-Bond Tasks-Tasks that are heavy on CPU usage(eg., mathematical computations,data processing)

import multiprocessing

import time

def square_numbers():
    for i in range(5):
        time.sleep(1,2)
        print(f"square:{i*i}")
        
def cube_numbers():
    for i in range(5):
        time.sleep(1,5)
        print(f"Cube:{i*i*i}")   

if __name__=="__main__":
        
    ## Create 2 Proceses
    p1=multiprocessing.Process(target=square_numbers)
    p2=multiprocessing.Process(target=cube_numbers)
    t=time.time()
    ## start the process 
    p1.start()
    p2.start()

    ## Wait for the process to complete
    p1.join()
    p2.join()        
                
    finished_time=time.time()-t
    print(finished_time)  
  
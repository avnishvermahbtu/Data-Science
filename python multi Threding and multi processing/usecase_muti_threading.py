'''
Real-word Example:Multitreading for I/O-bound Tasks
Scenario: Wed Scraping 
Wed scraping often involves making numerous network requests to
fetch wed pages.These tasks are I/O-bound because they spend a lot of
time watiting for response from servers .Multithreading can significantly
improve the performance by allowing mutiple web pages to fetched concurrently 

''' 
'''
https://python.langchain.com/v0.2/docs/introduction/

https://python.langchain.com/v0.2/docs/concepts/

https://python.langchain.com/v0.2/docs/tutorials/
'''

import threading
import requests
from bs4 import BeautifulSoup

urls=[
    
'https://python.langchain.com/v0.2/docs/introduction/',

'https://python.langchain.com/v0.2/docs/concepts/',

'https://python.langchain.com/v0.2/docs/tutorials/'
    
]

def fetch_content(url):
    # web page ko prapt karne ke liye get function
    response=requests.get(url)
    soup=BeautifulSoup(response.content,'html.parser')
    print(f'Fetched {len(soup.text)} characters from {url}')
    
threads=[]

for url in urls:
    thread=threading.Thread(target=fetch_content,args=(url,))
    threads.append(thread)
    thread.start()
    
for thread in threads:
    thread.join()

print("All wed pages feched")            
    








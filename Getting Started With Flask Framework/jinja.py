## Buliding Url Dynamically
## Variable Rule
## Jinja 2 Template Engine

### Jinja Template Engine 
'''
{{  }}    expressions to print output in html
{%...%}  condition,  for loops
{#...#}  this is for comments
'''


from flask import Flask,render_template,request,redirect,url_for

## render template use return
'''
It creates an instance of the flask class,
which will be your WSGI(Web Server Gateway Interface) application
'''

## WSGI Application
app=Flask(__name__)

@app.route("/")
def welcome():
    return"<html><H1>Welcome to the flask cource</H1></html>"

@app.route("/index",methods=['GET'])
def index():
     return render_template('index1.html')
 

@app.route('/sumit',methods=['GET','POST'])
def sumit():
     if request.method=='POST':
         name=request.form['name']
         return f'Hello {name}'
     return render_template('form.html') 

## Variable Rule 
@app.route('/success/<int:score>')
def success(score):
    res=""
    if score>=50:
        res="PASSED"
    else:
        res="FAILED"
        
    return render_template('result.html',results=res)    


## Variable Rule 
@app.route('/successres/<int:score>')
def successres(score):
    res=""
    if score>=50:
        res="PASSED"
    else:
        res="FAILED"
    
    exp={'score':score,"res":res}  ## exp is nothing but it is key value pair
        
    return render_template('result1.html',results=exp)    


## if confition 
@app.route('/successif/<int:score>')
def successif(score):
    res=""
    if score>=50:
        res="PASSED"
    else:
        res="FAILED"
        
    return render_template('result.html',results=res)    

@app.route('/fail/<int:score>')
def fail(score):
   
        
    return render_template('result.html',results=score)    

@app.route('/sumit',methods=['POST','GET'])
def sumit():
    total_score=0
    if request.method=='POST':
        science=float(request.form['science']) 
        maths=float(request.form['maths'])
        c=float(request.form['c'])
        data_science=float(request.form['datascience'])      
           
        total_score=(science+maths+c+data_science)/4   
    else:
        return render_template('getresult.html')    
    return  redirect(url_for('successres',score=total_score))     
           
           
if __name__=="__main__":
    app.run(debug=True,port=8000)




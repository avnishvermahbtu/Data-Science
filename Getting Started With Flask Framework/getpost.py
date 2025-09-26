from flask import Flask,render_template,request

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
 
@app.route('/form',methods=['GET','POST'])
def form():
     if request.method=='POST':
         name=request.form['name']
         return f'Hello {name}'
     return render_template('form.html')
 
@app.route('/sumit',methods=['GET','POST'])
def sumit():
     if request.method=='POST':
         name=request.form['name']
         return f'Hello {name}'
     return render_template('form.html') 

if __name__=="__main__":
    app.run(debug=True,port=8000)



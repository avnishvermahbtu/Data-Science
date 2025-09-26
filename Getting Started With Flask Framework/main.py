from flask import Flask,render_template

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

@app.route("/index") 
def index():
     return render_template('index1.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__=="__main__":
    app.run(debug=True,port=8000)



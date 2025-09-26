from flask import Flask

'''
It creates an instance of the flask class,
which will be your WSGI(Web Server Gateway Interface) application
'''

## WSGI Application
app=Flask(__name__)  #__name__ intry point of program

### Flask APP Url Routing

## Decorator
@app.route("/")
def welcome():
    return"Welcome to this  Flask course.This should be an amazing course"

@app.route("/index")   ## /index is Url
def index():
     return "Welcome to the index page"


if __name__=="__main__":
    app.run(debug=True)



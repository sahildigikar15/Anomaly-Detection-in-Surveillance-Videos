from flask import Flask, render_template,request,redirect,session,url_for,g
from datetime import datetime
import os

app = Flask(__name__)

@app.route("/")
def homePage():
    return render_template('index.html')

app.run(port=8181,debug=True)
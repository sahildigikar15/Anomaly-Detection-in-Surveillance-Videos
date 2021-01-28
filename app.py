from flask import Flask, render_template,request,redirect,session,url_for,g
import os
from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/final_year_project'
db = SQLAlchemy(app)
class User(db.Model):
    # sr,name,email,password
    sr = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50),nullable=False)
    email = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(15), nullable=False)

@app.route("/",methods = ['GET','POST'])
def home():
    return render_template('index.html')

@app.route("/homepage")
def homePage():
    return render_template('homePage.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/register")
def register():
    return render_template('register.html')
app.run(port=8181,debug=True)
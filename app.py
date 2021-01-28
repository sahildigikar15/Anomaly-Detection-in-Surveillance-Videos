from flask import Flask, render_template,request,redirect,session,url_for,g
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/final_year_project'
db = SQLAlchemy(app)
class User(db.Model):
    # sr,name,email,password
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50),nullable=False)
    email = db.Column(db.String(50), nullable=False)
    number = db.Column(db.String(12), nullable=False)
    password = db.Column(db.String(15), nullable=False)
    date = db.Column(db.String(12), nullable= True)

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

@app.route("/add_user",methods=['POST'])
def add_user():
    name = request.form.get('name')
    email = request.form.get('email')
    number = request.form.get('number')
    password = request.form.get('password')
    re_password = request.form.get('re_password')
    entry = User(name=name,number=number,email=email,date = datetime.now(),password=password)

    if password == re_password:

        db.session.add(entry)
        db.session.commit()
        return redirect('/')
    else:
        msg = "Password mismatch"
        return render_template('register.html',msg=msg)

@app.route("/login_validation",methods = ['POST'])
def login_validation():

    email = request.form.get('email')
    password = request.form.get('password')

    user_data = User.query.filter_by(email=email, password=password).first()
    try:
        if user_data.email is not None and user_data is not None:
            return redirect('/')
    except:
        msg = "Incorrect username/password!"
        return render_template('login.html',msg=msg)

app.run(port=8181,debug=True)
import glob

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, session, url_for, g, flash
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug import secure_filename
import mxnet as mx
from mxnet import gluon
from PIL import Image
import ml



app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://sql6412824:1ZjNmP9LIy@sql6.freemysqlhosting.net/sql6412824'

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploader')

app.config["IMAGE_UPLOADS"] = os.path.join('static', 'img_converted')
app.config["PRED_IMAGE"] = os.path.join('static', 'predicted_image')
app.config["fin_vid"] = os.path.join('static', 'final_video1')

db = SQLAlchemy(app)

app.secret_key = os.urandom(24)


class User(db.Model):
    # sr,name,email,password
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    number = db.Column(db.String(12), nullable=False)
    password = db.Column(db.String(15), nullable=False)
    date = db.Column(db.String(12), nullable=True)


@app.route('/dropsession')
def dropsession():
    session.pop('user', None)
    return render_template('index.html')


@app.before_request
def before_request():
    g.user = None
    if 'user' in session:
        g.user = session['user']


@app.route("/", methods=['GET', 'POST'])
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


@app.route("/add_user", methods=['POST'])
def add_user():
    name = request.form.get('name')
    email = request.form.get('email')
    number = request.form.get('number')
    password = request.form.get('password')
    re_password = request.form.get('re_password')
    entry = User(name=name, number=number, email=email, date=datetime.now(), password=password)

    if password == re_password:

        db.session.add(entry)
        db.session.commit()
        return redirect('/')
    else:
        msg = "Password mismatch"
        return render_template('register.html', msg=msg)


@app.route("/login_validation", methods=['POST'])
def login_validation():
    session.pop('user', None)
    email = request.form.get('email')
    password = request.form.get('password')

    user_data = User.query.filter_by(email=email, password=password).first()
    try:
        if user_data.email is not None and user_data.password is not None:
            session['user'] = user_data.name
            session['email'] = user_data.email
            return redirect('/user_dashboard')
    except:
        msg = "Incorrect username/password!"
        return render_template('login.html', msg=msg)


@app.route("/user_dashboard")
def user_dashboard():
    if g.user:
        user = session['user']
        email = session['email']

        return render_template('user_dashboard.html', user=user, email=email)
    else:
        return redirect('/login')


@app.route('/uploader')
def upload_form():
    if g.user:
        user = session['user']
        return render_template('multifiles.html',user = user)




@app.route("/uploader", methods=['POST'])
def uploader():
    if g.user:
        user = session['user']
        if 'files[]' not in request.files:
        #flash('No file part')
            return render_template(request.url)

        filelist = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(".tif")]
        for f in filelist:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

        filelist = [f for f in os.listdir(app.config["IMAGE_UPLOADS"]) if f.endswith(".jpeg")]
        for f in filelist:
            os.remove(os.path.join(app.config["IMAGE_UPLOADS"], f))


        files = request.files.getlist('files[]')
        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        for infile in os.listdir(app.config['UPLOAD_FOLDER']):
            if infile[-3:] == "tif" or infile[-3:] == "bmp":
                outfile = infile[:-3] + "jpeg"
                im = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], infile))
                    #app.config['UPLOAD_FOLDER'] + infile)
                out = im.convert("RGB")
                save_path = os.path.join(app.config['IMAGE_UPLOADS'],outfile )
                   # app.config["IMAGE_UPLOADS"] + outfile
                out.save(save_path, "JPEG", quality=90)



        filelist = [os.path.join(app.config['IMAGE_UPLOADS'], f) for f in os.listdir(app.config["IMAGE_UPLOADS"]) if f.endswith(".jpeg")]
        print(filelist)
        return render_template('multifiles.html',user = user,data= filelist)

@app.route('/stores')
def store_locate():
    if g.user:
        user = session['user']
        return render_template('index_map.html',user = user)


@app.route('/detect_anomalies')
def find_anomaly():
    if g.user:
        user = session['user']
        print("1] Reading Data and Preprocessing. 2] Loading the Model. 3] Predicting ")
        test_file = sorted(glob.glob(os.path.join('static', 'uploader/*')))
        a = np.zeros((len(test_file), 2, 100, 100))
        for idx, filename in enumerate(test_file):
            im = Image.open(filename)
            im = im.resize((100, 100))
            a[idx, 0, :, :] = np.array(im, dtype=np.float32) / 255.0

        dataset = gluon.data.ArrayDataset(mx.nd.array(a, dtype=np.float32))
        dataloader = gluon.data.DataLoader(dataset, batch_size=1)

        print("Done Reading and preprocessing")

        model = ml.ConvolutionalAutoencoder()
        model.load_parameters(os.path.join('static', 'autoencoder_ucsd.params'))
        reg_scores_cae = ml.plot_regularity_score(model, dataloader)

        print("Done loading params")

        filelist = [f for f in os.listdir(app.config['PRED_IMAGE']) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(app.config['PRED_IMAGE'], f))


        ml.model_evaluation(model,dataloader)
        print("Done Predicting and now showing predicted frames !!")



        filelist = [os.path.join(app.config['PRED_IMAGE'], f) for f in os.listdir(app.config["PRED_IMAGE"]) if f.endswith(".png")]


        return render_template('predicted_images.html',user = user,data= filelist)


@app.route('/convert_video')
def convert_video():
    if g.user:
        user = session['user']
        if os.path.exists(os.path.join(app.config["fin_vid"] , 'project.mp4')):
            os.remove(os.path.join(app.config["fin_vid"], 'project.mp4'))

        img_array = []
        for filename in sorted(glob.glob(os.path.join('static', 'predicted_image/*'))):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter('static/final_video1/project.mp4', cv2.VideoWriter_fourcc(*'webm'), 15, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        return render_template('final_video.html', user=user)

app.run(port=8181, debug=True)

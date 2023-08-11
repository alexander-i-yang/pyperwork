from flask import Flask, render_template, send_from_directory, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

import os
import json

import filler

UPLOAD_FOLDER = './pdfs/'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def homepage():
    if request.method == "GET":
        return render_template("index.html", title="HOME PAGE")
    # elif request.method == "POST":
    #     uploaded_file = request.files['file']
    #     if uploaded_file.filename != '':
    #         uploaded_file.save("./pdfs/" + uploaded_file.filename)

    if request.method == 'POST':
        # check if the post request has the file part
        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print("Yey1")
            filename = secure_filename(file.filename)
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(url_for('download_file', name=filename))
            return redirect(url_for('download_file', name=filename))
    return ''

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
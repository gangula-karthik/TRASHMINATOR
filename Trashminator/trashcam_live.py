from re import DEBUG, sub
from flask import Flask, render_template, Response, request, redirect, send_file, url_for
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess

app = Flask(__name__)

    

@app.route('/')
def index():
    return render_template('trashcam_live.html')

@app.route("/opencam", methods=['GET'])
def opencam():
    print("here")
    subprocess.call(["python3", "detect.py", "--source", "0", "--weights", "yolov7_files/epoch_054.pt", "--conf","0.25"])
    return Response(opencam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=8000)
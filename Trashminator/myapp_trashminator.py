from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from trash_detector import detectorTrash
import os
app = Flask(__name__)

# @app.route('/upload')
# def upload_file():
#    return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file_upload']
      f.save(secure_filename(f.filename))
      file_path = os.path.abspath(f.filename)
      return detectorTrash(file_path)

if __name__ == '__main__':
   app.run(debug = True, port=5000)



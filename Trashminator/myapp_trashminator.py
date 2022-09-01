from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from trash_detector import detectorTrash
app = Flask(__name__)

# @app.route('/upload')
# def upload_file():
#    return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file_upload']
      f.save(secure_filename(f.filename))
      if detectorTrash(f'/Users/karthik/Documents/GitHub/Codeoverflow_nyp/Trashminator/{f.filename}') == True:
         return 'The video has been uploaded successfully'
      else:
         return 'Try again :('
		
if __name__ == '__main__':
   app.run(debug = True)
from flask import *  
import underwaterimageprocessing as u
#import low as u
app = Flask(__name__)  
 
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/complete', methods = ['GET','POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        print ("file uploaded successfully")
        u.mainFunction(f.filename)
        print("Calling mainFunction")
	
    if request.method == 'GET':
        f = request.files['file']  
        f.save(f.filename)  
        print ("file uploaded successfully")
        u.mainFunction(f.filename)
        print("Calling mainFunction")
        
    return render_template("file_upload_form.html") 
'''@app.route("/upload")
def send_image():
	filename="out.png"
	return send_from_directory("static",filename)'''
  
if __name__ == '__main__':
    app.run(threaded=True)  
    app.run(debug = True)  

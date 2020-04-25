from flask import *  
import underwaterimageprocessing as u
#import low as u
app = Flask(__name__)  
 
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        return render_template("success.html", name = f.filename)  


@app.route('/runSpeedFunction',methods = ['GET', 'POST'])  
def runSpeedFunction():  
    if request.method == 'POST':
        u.mainFunction()
        print("Calling mainFunction")
    if request.method == 'GET':
        u.mainFunction()
        print("Calling mainFunction")
        
    return render_template("file_upload_form.html") 
  
if __name__ == '__main__':
    app.run(threaded=True)  
    app.run(debug = True)  

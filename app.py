from scripts.caller import MalariaDetector,predict_malaria
from scripts.config import Config as cfg 
import os,io
from PIL import Image
from flask import Flask,request,jsonify,render_template


app=Flask(__name__)
@app.route('/')
def welcome():
    return render_template('index.html')
@app.route('/predict_malaria',methods=['POST'])
def malaria_detect_route():
    try:
        image_f=request.files['image']
        img_data = Image.open(io.BytesIO(image_f.read()))
        print(img_data)
        output = predict_malaria(img_data)
        return render_template('result.html',result=output)
    except Exception as e:
        return jsonify({"error":str(e)})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int("5000"),debug=True)




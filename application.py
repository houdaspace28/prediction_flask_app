from flask import Flask, render_template, request,jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

application = Flask(__name__)
app = application

model= pickle.load(open('models/linear_regression.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data',methods=['POST','GET'])
def predict_data():
    if request.method == 'POST':
        Temperature= float(request.form.get('Temperature'))
        RH= float(request.form.get('RH'))
        Ws= float(request.form.get('Ws'))
        Rain= float(request.form.get('Rain'))
        FFMC= float(request.form.get('FFMC'))
        DMC= float(request.form.get('DMC'))
        ISI= float(request.form.get('ISI'))
        Classes= float(request.form.get('Classes')) 
        Region= float(request.form.get('Region'))

        scaled_data = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = model.predict(scaled_data)
        return render_template('home.html',results=result[0])
    else:
        render_template('home.html')


if  __name__=='__main__':
    app.run(host='0.0.0.0')
import pickle
from flask import Flask, request, jsonify, app, url_for_render, render_template
import pandas as pd
import numpy as np

app=Flask(__name__)
## Load imputer
imputer=pickle.load(open('imputer.pkl','rb'))
## Load scaler
scaler=pickle.load(open('scaler.pkl','rb'))
## Load encoder
encoder=pickle.load(open('label_encoder.pkl','rb'))
## Load model 1
rfmodel=pickle.load(open('rf_vehicle.pkl','rb'))
## Load model 2
meta_model=pickle.load(open('meta_vehicle.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict():
    data=request.get_json(force=True)
    data=pd.DataFrame(data)
    data=imputer.transform(data)
    data=scaler.transform(data)
    prediction=rfmodel.predict(data)
    prediction=encoder.inverse_transform(prediction)
    return jsonify({'prediction':prediction.tolist()})
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 11:45:03 2021

@author: noopa
"""


import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
pickle_in = open("gb_clf.pkl","rb")
classifier=pickle.load(pickle_in)
# classifier=pickle.load(open("gb_clf.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(final_features)
    output = prediction[0]

    
    return render_template('index.html', prediction_text='The fish belong to species {}'.format(output))
    
    


if __name__=='__main__':
    app.run(debug=True)
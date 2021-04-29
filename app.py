from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from numpy.lib.arraysetops import setxor1d


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
@app.route('/home')
def Home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    if output == 1:
        return render_template('index.html',prediction_text = 'High Chances of Heart Attack')
    else:
        return render_template('index.html',prediction_text = 'Low Chances of Heart Attack')
        
if __name__ == "__main__":
    app.run(debug=True)
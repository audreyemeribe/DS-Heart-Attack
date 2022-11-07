# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the XGBoost Classifier model
model = pickle.load(open('pickled_best_XGB.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('main.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trtbps = int(request.form['trtbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalachh = int(request.form['thalachh'])
        exng = request.form.get('exng')
        oldpeak = float(request.form['oldpeak'])
        slp = request.form.get('slp')
        caa = int(request.form['caa'])
        thall = request.form.get('thall')

        data = np.array([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]])
        my_prediction = model.predict(data)

        return render_template('result.html', prediction=my_prediction)



if __name__ == '__main__':
  app.run(debug=True)

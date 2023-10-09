# to control all routes

from flask import Flask, render_template, request
import carpriceprediction as a1
from a3 import fn_a3_predict
from a2carpriceprediction import *;

app = Flask(__name__)

filename = '/root/source_code/app/values.pkl'
values_class = pickle.load(open(filename,'rb'))

# Setting app to run on Flask
ohe = values_class['ohe']

# call landing page
@app.route('/')
def hello_world():
    return render_template('index.html')

# call old prediction page
@app.route('/a1_carpriceprediction')
def a1_carpriceprediction():
    return render_template('old_model_prediction.html')

# call new prediction page
@app.route('/a2_carpriceprediction')
def a2_carpriceprediction():
    return render_template('new_model_prediction.html')

# call classification page
@app.route('/a3_carpriceclassification')
def a3_carpriceprediction():
    return render_template('regularization_model.html',brands = ohe.categories_[0])

# predict using old model
@app.route('/a1_predict',methods=['POST'])
def a1_predict():
    a1_input_features = [float(request.form['mileage']),
                         float (request.form['year']),
                         float(request.form['brand'])]
    prediction = a1.fn_a1_predict(a1_input_features)
    return render_template('old_model_prediction.html',prediction=prediction)

# predict using new model
@app.route('/a2_predict',methods=['POST'])
def a2_predict():
    a2_input_features = [float(request.form['mileage']),
                         float (request.form['year']),
                         float(request.form['brand'])]
    prediction = fn_a2_predict(a2_input_features)
    prediction = np.exp(prediction)
    return render_template('new_model_prediction.html',prediction=prediction)

# predict using classification model
# @app.route('/a3_predict',methods=['POST'])
# def a3_predict():
#     a3_input_features = [float(request.form['mileage']),
#                          float (request.form['year']),
#                          float(request.form['brand'])]
#     prediction = fn_a3_predict(a3_input_features)
#     prediction = np.exp(prediction)
#     return render_template('regularization_model.html',prediction=prediction)

# The route to calculate the prediction result but not accessed by users
@app.route('/a3_predict', methods = ['POST'])
def a3_predict():
    if request.method == 'POST':
        # Getting the values required for prediction
        brand_name = request.form.get('brand')
        # brand = list(ohe.transform([[brand_name]]).toarray()[0])
        fuel = request.form.get('fuel')
        year = request.form.get('year')
        max_power = request.form.get('max_power')
        print(brand_name)
        
        a3_input_features = [brand_name, fuel, year, max_power]
        # print(a3_input_features)

        # Calling the prediction function, coverting the result to int for user experience and then to string
        prediction = fn_a3_predict(a3_input_features)
        # to display on the website
        # result = prediction_class(name,max,year,fuel)

        return render_template('regularization_model.html',prediction=prediction, brands = ohe.categories_[0])



if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port=80)
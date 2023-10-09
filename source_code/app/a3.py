import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import pickle
import mlflow

# modelname = 'source_code/app/CarPricePrediction.model'
# |--- This part need to store the model in mlflow --|
# Loading the default values and classification model

# what is it???
filename = 'values.pkl'
values_class = pickle.load(open('/root/source_code/app/values.pkl','rb'))

# # load the classification with Regression using Ridge model with hyper parmeter
# mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
# model_name = "st124087-a3-model"
# model_class = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")

# # load the classification with Normal Regression : st124087-a3-model-NormalLogisticReg
# mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
# model_name = "st124087-a3-model-NormalLogisticReg"
# model_class = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")

# load for the requied data format
filename = '/root/source_code/app/values.pkl'
values_class = pickle.load(open(filename,'rb'))

# load the mlflow model
mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
model_name = "st124087-a3-model"
model_class = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")

# Setting app to run on Flask
k_range = values_class['k_range']
scaler_class = values_class['scaler']
ohe = values_class['ohe']

# Classification function to predit car price
def fn_a3_predict (input):
    
    brand_name= input[0]
    fuel = input[1]
    year = input[2]
    max_power = input[3]
    
    brand = list(ohe.transform([[brand_name]]).toarray()[0])
    print('one hot brand', brand)
    max_power = 40
    sample = np.array([[max_power,year,fuel]+brand])
        
    # Scale the input data using the trained scaler
    sample[:, 0: 2] = scaler_class.transform(sample[:, 0: 2])
    sample = np.insert(sample, 0, 1, axis=1)
    
    expected_feature_names = ['intercept','max_power', 'year', 'fuel', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']

    # Set the column names of the 'sample' DataFrame
    sample = pd.DataFrame(sample, columns=expected_feature_names)
    
    # Turn the input data into polynomial
    # Predict the car price using the trained model
    result = model_class.predict(sample)

    # Return the price range of the car based on the predicted class
    return k_range[result[0]]
    
print("*****  successfully called car price prediction and parse the predicted value *****")
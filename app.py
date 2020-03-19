import os
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
#%matplotlib inline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC  
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin
from sklearn import preprocessing

app = Flask(__name__, static_url_path='/static')
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

def predict(to_predict_list):

    df = pd.read_csv('hotel_bookings.csv')
    #divide features into numeric and categorical
    #num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",
     #           "stays_in_weekend_nights","stays_in_week_nights","adults","children",
      #          "babies","is_repeated_guest", "previous_cancellations",
       #         "previous_bookings_not_canceled","agent","company",
        #        "required_car_parking_spaces", "total_of_special_requests", "adr"]
    cat_features = ["hotel","arrival_date_month","meal","market_segment",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"]
    # Separate features and predicted value
    features =cat_features
    X = df.drop(["is_canceled"], axis=1)[features]
    y = df["is_canceled"]

    #cat_features = ["hotel","arrival_date_month","meal","market_segment",
     #           "distribution_channel","reserved_room_type","deposit_type","customer_type"]


    #you could do something like one-hot-encoding of data_categorical here

    le = preprocessing.LabelEncoder()
    #data_categorical = data_categorical.apply(le.fit_transform)

    mapping_dict={}
    for col in X:
        X[col] = le.fit_transform(X[col])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        mapping_dict[col]=le_name_mapping

    #print(mapping_dict)
    #join the two masked dataframes back together
   # X= pd.concat([data_numeric, data_categorical], axis = 1)
    #X=X.drop(["is_canceled"], axis=1)
        
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X,y)
        
        #Predict form data
    to_predict_array = np.array(to_predict_list)
    to_predict_array=to_predict_array.reshape(1, -1)
    result = clf.predict(to_predict_array)


  
    return result[0]
    
@app.route('/final',methods=['GET','POST'])
def final():

    data = request.get_json()
    
    print (data['hotel'])
    hotel=data['hotel']
    arrival_month=data['arrival_month']
    meal=data['meal']
    MS=data['MS']
    DC=data['DC']
    r_room_type=data['r_room_type']
    deposit_type=data['deposit_type']
    customer_type=data["customer_type"]
    
    
    # # #put everything into int
    hotel=int(hotel)
    arrival_month=int(arrival_month)
    meal=int(meal)
    MS=int(MS)
    DC=int(DC)
    r_room_type=int(r_room_type)
    deposit_type=int(deposit_type)
    customer_type=int(customer_type)
    
    to_predict_list = [hotel, arrival_month,meal,MS,DC,r_room_type,deposit_type,customer_type]
    result = predict(to_predict_list)   
    result=int(result)

    return jsonify({ 
        'result': result
    })


if __name__== '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

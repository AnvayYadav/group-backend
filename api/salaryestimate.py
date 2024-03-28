from flask import Flask, request, jsonify
from flask import Blueprint
from flask_restful import Api, Resource
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

salary_estimate_api = Blueprint('salary_estimate', __name__, url_prefix='/api/salary_estimate')
api = Api(salary_estimate_api)

class SalaryEstimateAPI(Resource):
    def __init__(self):
        salary_estimate_data = pd.read_csv('/.../static/jobs.csv')
        td = salary_estimate_data.copy()
        td.dropna(inplace=True)
        td['_title'] = td['_title'].apply(lambda x: 1 if x == 'Software Engineer' else 0)
        td['_qualification'] = td['_qualification'].apply(lambda x: 1 if x == "Bachelors" else (2 if x == "Masters" else 0)) #0 is pHd
        td['_field'] = td['_field'].apply(lambda x: 1 if x else 0)

        self.enc = OneHotEncoder(handle_unknown='ignore')
        embarked_encoded = self.enc.fit_transform(td[['embarked']].values.reshape(-1, 1))
        self.encoded_cols = self.enc.get_feature_names_out(['embarked'])

        td[self.encoded_cols] = embarked_encoded.toarray()
        td.drop(['embarked'], axis=1, inplace=True)

        self.logreg = LogisticRegression(max_iter=1000)
        X = td.drop('salary_estimate', axis=1)
        y = td['salary_estimate']
        self.logreg.fit(X, y)

    def predict_salary(self, data):
        try:
            person = pd.DataFrame([data]) 
            person['_title'] = person['_title'].apply(lambda x: 1 if x == 'Sotware Engineer' else 0)
            person['_'] = person['alone'].apply(lambda x: 1 if x else 0)


            dead_proba, alive_proba = np.squeeze(self.logreg.predict_proba(person))

            return {
                'Salary Estimate': '{:100}'.format(dead_proba, alive_proba), #idk if this will work??
            }
        except Exception as e:
            return {'error': str(e)}


    def post(self):
        try:
            data = request.json
            result = self.predict_salary(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})

api.add_resource(SalaryEstimateAPI, '/predict')
















# from flask import Flask, request, jsonify, Blueprint
# from flask_restful import Api, Resource
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier

# # Define the Blueprint for the Salary Estimate API
# salary_estimate_api = Blueprint('salary_estimate_api', __name__, url_prefix='/api/salary_estimate')
# api = Api(salary_estimate_api)

# class SalaryEstimateAPI(Resource):
#     def __init__(self):
#         # Load the heart disease dataset
#         pay_data = pd.read_csv('/../static/jobs.csv/')

#         # Perform data preprocessing
#         X = pay_data.drop('_pay', axis=1)
#         y = pay_data['_pay']
#         self.scaler = StandardScaler()
#         X_scaled = self.scaler.fit_transform(X)

#         # Initialize the Random Forest classifier
#         self.rf_classifier = RandomForestClassifier()

#         # Train the classifier
#         self.rf_classifier.fit(X_scaled, y)

#     def predict_salary_estimate(self, data):
#         try:
#             # Create a DataFrame from the input data
#             input_data = pd.DataFrame([data])

#             # Scale the input data using the same scaler used during training
#             input_scaled = self.scaler.transform(input_data)

#             # Predict the likelihood of heart disease
#             prediction = self.rf_classifier.predict_proba(input_scaled)[:, 1]

#             return {'Estimated Salary': prediction[0]}
#         except Exception as e:
#             return {'error': str(e)}

#     def post(self):
#         try:
#             data = request.json
#             result = self.predict_salary_estimate(data)
#             return jsonify(result)
#         except Exception as e:
#             return jsonify({'error': str(e)})

# api.add_resource(SalaryEstimateAPI, '/predict')
from flask import Flask, request, jsonify
from flask import Blueprint
from flask_restful import Api, Resource
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np

lungcancerprediction_api = Blueprint('lungcancerprediction_api', __name__, url_prefix='/api/lungcancerprediction')
api = Api(lungcancerprediction_api)

class CancerPredictionAPI(Resource):
     def __init__(self):
        Cancer_prediction_data = pd.read_csv('/.../static/surveylungcancer.csv')
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


            def __init__(self):
        titanic_data = sns.load_dataset('titanic')
        td = titanic_data.copy()
        td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        td.dropna(inplace=True)
        td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
        td['alone'] = td['alone'].apply(lambda x: 1 if x else 0)

        self.enc = OneHotEncoder(handle_unknown='ignore')
        embarked_encoded = self.enc.fit_transform(td[['embarked']].values.reshape(-1, 1))
        self.encoded_cols = self.enc.get_feature_names_out(['embarked'])

        td[self.encoded_cols] = embarked_encoded.toarray()
        td.drop(['embarked'], axis=1, inplace=True)

        self.logreg = LogisticRegression(max_iter=1000)
        X = td.drop('survived', axis=1)
        y = td['survived']
        self.logreg.fit(X, y)

    def predict_survival(self, data):
        try:
            passenger = pd.DataFrame([data]) 
            passenger['sex'] = passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
            passenger['alone'] = passenger['alone'].apply(lambda x: 1 if x else 0)

            embarked_encoded = self.enc.transform(passenger[['embarked']].values.reshape(-1, 1))
            passenger[self.encoded_cols] = embarked_encoded.toarray()
            passenger.drop(['embarked', 'name'], axis=1, inplace=True)

            dead_proba, alive_proba = np.squeeze(self.logreg.predict_proba(passenger))

            return {
                'Death probability': '{:.2%}'.format(dead_proba),
                'Survival probability': '{:.2%}'.format(alive_proba)
            }
        except Exception as e:
            return {'error': str(e)}


    def post(self):
        try:
            data = request.json
            result = self.predict_survival(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})

api.add_resource(TitanicAPI, '/predict')
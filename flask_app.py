from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        train_df = pd.read_csv('./train_df.csv')
        gender = request.form['gender']
        age = request.form['age']
        family_history = request.form['family_history']
        benefits = request.form['benefits']
        care_options = request.form['options']
        anonymity = request.form['anonymity']
        leave = request.form['leave']
        work_interfere = request.form['interferes']
        df = pd.DataFrame({
            'Gender': gender,
            'Age': int(age),
            'family_history': family_history,
            'benefits': benefits,
            'care_options': care_options,
            'anonymity': anonymity,
            'leave': leave,
            'work_interfere': work_interfere,
            'treatment': 'No'
        }, index=[0])

        le_gender = preprocessing.LabelEncoder()
        train_df['Gender'] = le_gender.fit(train_df['Gender'])
        le_family = preprocessing.LabelEncoder()
        train_df['family_history'] = le_family.fit(train_df['family_history'])
        le_benefits = preprocessing.LabelEncoder()
        train_df['benefits'] = le_benefits.fit(train_df['benefits'])
        le_options = preprocessing.LabelEncoder()
        train_df['care_options'] = le_options.fit(train_df['care_options'])
        le_anon = preprocessing.LabelEncoder()
        train_df['anonymity'] = le_anon.fit(train_df['anonymity'])
        le_leave = preprocessing.LabelEncoder()
        train_df['leave'] = le_leave.fit(train_df['leave'])
        le_inter = preprocessing.LabelEncoder()
        train_df['work_interfere'] = le_inter.fit(train_df['work_interfere'])
        le_treat = preprocessing.LabelEncoder()
        train_df['treatment'] = le_treat.fit(train_df['treatment'])

        df['Gender'] = le_gender.transform(df['Gender'])
        df['family_history'] = le_family.transform(df['family_history'])
        df['benefits'] = le_benefits.transform(df['benefits'])
        df['care_options'] = le_options.transform(df['care_options'])
        df['anonymity'] = le_anon.transform(df['anonymity'])
        df['leave'] = le_leave.transform(df['leave'])
        df['work_interfere'] = le_inter.transform(df['work_interfere'])
        df['treatment'] = le_treat.transform(df['treatment'])

        # Scaling Age
        scaler = preprocessing.MinMaxScaler()
        train_df['Age'] = scaler.fit_transform(train_df[['Age']])

        # Scale new data
        df['Age'] = scaler.transform(df[['Age']])

        # define X and y
        feature_cols = ['Age', 'Gender', 'family_history', 'benefits',
                        'care_options', 'anonymity', 'leave', 'work_interfere']
        X = df[feature_cols]

        loaded_model = pickle.load(open('./stack_model.sav', 'rb'))
        preds = loaded_model.predict(X)
        inverted_predictions = le_treat.inverse_transform(preds)
        return render_template('results.html', preds=inverted_predictions)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

import pickle

from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)


@app.route("/")
def index():
    cat_cols = {
        'IsFirstTime': ['n', 'y'],
        'Occupancy': ['o', 'i', 's'],
        'Channel': ['t', 'r', 'c', 'b'],
        'PPM': ['n', 'y'],
        'ProductType': ['frm'],
        'PropertyState': ['il', 'co', 'ks', 'ca', 'nj', 'wi', 'fl', 'ct', 'ga', 'tx', 'md',
                          'ma', 'sc', 'wy', 'nc', 'az', 'in', 'ms', 'ny', 'wa', 'ar', 'va',
                          'mn', 'la', 'pa', 'or', 'ri', 'ut', 'mi', 'tn', 'al', 'mo', 'ia',
                          'nm', 'nv', 'oh', 'ne', 'vt', 'hi', 'id', 'pr', 'dc', 'gu', 'ky',
                          'nh', 'sd', 'me', 'mt', 'ok', 'wv', 'de', 'nd', 'ak'],
        'PropertyType': ['sf', 'pu', 'co', 'mh', 'cp', 'lh'],
        'LoanPurpose': ['p', 'n', 'c'],
        'hasMIP': ['True', 'False']
    }

    num_cols = {
        'FirstPaymentDate': 0,
        'OrigUPB': 117000,
        'OrigInterestRate': 6.75,
        'OrigLoanTerm': 360,
        'EverDelinquent': 0,
        'MonthsDelinquent': 0,
        'CreditRange': 1,
        'LTV_Range': 3,
        'Repay_Range': 2,
        'Loan_Duration': 2999,
        'Total_Interest_Paid': 284310000.0,
        'NumBorrowers': 2,
    }

    return render_template("index.html", cat_cols=cat_cols, num_cols=num_cols)


@app.route("/", methods=['POST'])
def predict():
    prediction = ""
    message = ""
    if request.method == 'POST':
        model_filename = 'pre-trained-model/pipeline.pkl'
        with open(model_filename, 'rb') as model_file:
            pipeline = pickle.load(model_file)

        data = {}
        for variable in ['FirstPaymentDate', 'IsFirstTime', 'MaturityDate', 'MIP', 'Occupancy',
                         'OCLTV', 'DTI', 'OrigUPB', 'OrigInterestRate', 'Channel', 'PPM',
                         'ProductType', 'PropertyState', 'PropertyType', 'LoanPurpose',
                         'OrigLoanTerm', 'NumBorrowers', 'EverDelinquent', 'MonthsDelinquent',
                         'CreditRange', 'LTV_Range', 'Repay_Range', 'Loan_Duration',
                         'Total_Interest_Paid', 'hasMIP']:
            data[variable] = request.form.get(variable)

        data["hasMIP"] = data["hasMIP"] == "True"

        sample = pd.DataFrame([data])

        message = "Predicting the Prepayment Risk Ratio ..."
        prediction = pipeline.predict(sample)[0] * 100
    return render_template('results.html', prediction=prediction, message=message)


@app.route("/tr", methods=['POST'])
def train():
    data = pd.read_csv("data/data.csv")

    X = data
    y = data.pop("PPR")

    cat_cols = data.select_dtypes("object").columns
    num_cols = data.select_dtypes(exclude="object").columns

    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # pipelines categorical variables
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant")),
            ("one_hot_encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # pipeline of num variables
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean"))
        ]
    )

    # preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols),
        ]
    )

    # now let's build our main pipeline called (bai pipeline)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=27)),
            ("model", RandomForestRegressor()),
        ]
    )

    message = ""
    if request.method == 'POST':
        model_filename = 'pre-trained-model/pipeline.pkl'
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)

        with open(model_filename, 'wb') as model_file:
            pickle.dump(pipeline, model_file)

        message = f"Here is the accuracy : {accuracy}"

    return render_template('results.html', prediction="", message=message)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)

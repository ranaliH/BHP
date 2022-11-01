from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))
dummy = pickle.load(open('dummy.pkl', 'rb'))
X = pd.read_pickle("dummy.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    location1 = request.form['location']
    sqf1 = request.form['Total square feet']
    bathrooms1 = request.form['No_bathrooms']
    bhk1 = request.form['bhk']
    print(location1)

    def predict_price(location, sqft, bath, bhk):
        loc_index = np.where(X.columns == location)[0][0]

        x = np.zeros(len(X.columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1
        return model.predict([x])[0]

    # prediction = model.predict([[location, sqf, bathrooms, bhk]])

    # output = round(prediction[0], 2)
    output = predict_price(location1, sqf1, bathrooms1, bhk1)
    return render_template('index.html', prediction_text='House price would be Indian LKS {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

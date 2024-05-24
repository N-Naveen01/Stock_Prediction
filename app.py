from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load your CSV data
df = pd.read_csv("C:\\Users\\nnave\\Documents\\Design Studio\\Share Market\\laopala.csv")


# Split the dataset into features (X) and target variable (y)
X = df[['Open', 'High', 'Low']]  # Features
y = df['Close']                   # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your machine learning model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])

    # Prepare the input data for prediction
    input_data = np.array([[open_price, high_price, low_price]])

    # Get the predicted close price
    predicted_close_price = rf_regressor.predict(input_data)[0]

    return render_template('index.html', predicted_close_price=predicted_close_price)

if __name__ == '__main__':
    app.run(debug=True)

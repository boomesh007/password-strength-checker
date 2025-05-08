from flask import Flask, request, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import math
from collections import Counter

app = Flask(__name__)

# Load model and scaler
model = load_model("final_password_strength_model.h5")
scaler = joblib.load("scaler.pkl")

# Feature extraction
def extract_features(password):
    def password_length(p): return len(p)
    def count_uppercase(p): return sum(1 for c in p if c.isupper())
    def count_lowercase(p): return sum(1 for c in p if c.islower())
    def count_digits(p): return sum(1 for c in p if c.isdigit())
    def count_special(p): return sum(1 for c in p if not c.isalnum())
    def has_uppercase(p): return int(any(c.isupper() for c in p))
    def has_lowercase(p): return int(any(c.islower() for c in p))
    def has_digit(p): return int(any(c.isdigit() for c in p))
    def has_special(p): return int(any(not c.isalnum() for c in p))
    def character_variety(p): return has_uppercase(p) + has_lowercase(p) + has_digit(p) + has_special(p)
    def calculate_entropy(p):
        if not p: return 0
        counts = Counter(p)
        probs = [c / len(p) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    return np.array([[
        password_length(password),
        count_uppercase(password),
        count_lowercase(password),
        count_digits(password),
        count_special(password),
        has_uppercase(password),
        has_lowercase(password),
        has_digit(password),
        has_special(password),
        character_variety(password),
        calculate_entropy(password)
    ]])

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        password = request.form['password']
        features = extract_features(password)
        features_scaled = scaler.transform(features)
        pred = np.argmax(model.predict(features_scaled), axis=1)[0]
        prediction = {0: "Weak", 1: "Medium", 2: "Strong"}[pred]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Load model and features once at the start
model = joblib.load("models/ames_model.pkl")
FEATURES = json.load(open("features.json"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_file", methods=["POST"])
def predict_file():
    try:
        # Read uploaded CSV file
        f = request.files.get("file")
        df = pd.read_csv(f)

        # Drop target column if present
        df = df.drop(columns=["SalePrice"], errors="ignore")

        # Align with training features
        df = df.reindex(columns=FEATURES, fill_value=0)

        # Predict
        preds = model.predict(df)

        # Show first 5 predictions
        return render_template("index.html",
                               result=[float(x) for x in preds[:5]],
                               error=None)

    except Exception as e:
        return render_template("index.html", result=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)





    

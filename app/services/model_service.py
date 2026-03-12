import joblib
import pandas as pd

bundle = joblib.load("model/model_bundle.pkl")

class AttritionModelService:

    def __init__(self):
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.columns = bundle["features"]

    def preprocess(self, data):
        df = pd.DataFrame([data.model_dump()])
        df = df.reindex(columns=self.columns, fill_value=0)
        df_scaled = self.scaler.transform(df)

        return df_scaled

    def predict(self, data):

        X = self.preprocess(data)

        prob = self.model.predict_proba(X)[0][1]
        label = "Yes" if prob >= 0.5 else "No"

        return label, float(prob)
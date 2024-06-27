################## API

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd


model = joblib.load("pipeline_with_lgbm.pkl")


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
        gender = request.form.get("gender")
        senior = request.form.get("senior")
        partner = request.form.get("partner")
        dependents = request.form.get("dependents")
        tenure = request.form.get("tenure")
        phone_service = request.form.get("phone service")
        multi_lines = request.form.get("multiple lines")
        internet_service = request.form.get("internet service")
        online_security = request.form.get("online security")
        online_backup = request.form.get("online backup")
        device_protection = request.form.get("device protection")
        tech_support = request.form.get("tech support")
        streaming_tv = request.form.get("streaming tv")
        streaming_movies = request.form.get("streaming movies")
        contract = request.form.get("contract")
        paperless_billing = request.form.get("paperless billing")
        payment_method = request.form.get("payment method")
        month_charge = request.form.get("monthly charge")
        total_charge = request.form.get("total charge")


        result = {"gender": gender,
                  "senior": senior,
                  "partner": partner,
                  "dependents": dependents,
                  "tenure": tenure,
                  "phone service": phone_service,
                  "multiple lines": multi_lines,
                  "internet service": internet_service,
                  "online security": online_security,
                  "online backup": online_backup,
                  "device protection": device_protection,
                  "tech support": tech_support,
                  "streaming tv": streaming_tv,
                  "streaming movies": streaming_movies,
                  "contract": contract,
                  "paperless billing": paperless_billing,
                  "payment method": payment_method,
                  "monthly charge": month_charge,
                  "total charge": total_charge}

        result_list = []
        for i in result.values():
            result_list.append(i)

        data = np.array(result_list).reshape(1, -1)

        df_query = pd.DataFrame(data=data, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])


        pred = model.predict(df_query)[0]

        return jsonify({"Churn": str(pred)})


if __name__ == "__main__":
    app.run(debug=True)

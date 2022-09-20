import pickle
import numpy as np
from flask import Flask, render_template, request

# Load the Random Forest Classifier Bike model
filename1 = "Bike_Price_ML_Model_main.pkl"
classifier1 = pickle.load(open(filename1, "rb"))

# Load the Random Forest Classifier Car model
filename2 = "Car_Price_ML_Model.pkl"
classifier2 = pickle.load(open(filename2, "rb"))


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about", methods=["GET", "POST"])
def about():
    return render_template("about.html")


@app.route("/Bike", methods=["GET", "POST"])
def Bike():
    return render_template("Bike.html")


@app.route("/Car", methods=["GET", "POST"])
def Car():
    return render_template("Car.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    
    # Bike Prediction--------------------------------------------------------------
    if request.method == "POST":
        if len([float(x) for x in request.form.values()]) == 6:
            seller_type = int(request.form["seller_type"])
            owner = int(request.form["owner"])
            km_driven = int(request.form["km_driven"])
            ex_showroom_price = float(request.form["ex_showroom_price"])
            brand = int(request.form["brand"])
            no_of_yr = int(request.form["no_of_yr"])
            no_of_yr = 2021 - no_of_yr

            data = np.array(
                [[seller_type, owner, km_driven, ex_showroom_price, brand, no_of_yr]]
            )
            my_prediction = classifier1.predict(data)
            return render_template("result.html", prediction_text=round(my_prediction[0],2))

        # Car Prediction-------------------------------------------------------
        else:
            Fuel_Type_Diesel = 0
            Year = int(request.form["Year"])
            Present_Price = float(request.form["Present_Price"])
            Kms_Driven = int(request.form["Kms_Driven"])
            Owner = int(request.form["Owner"])
            Fuel_Type_Petrol = request.form["Fuel_Type_Petrol"]
            if Fuel_Type_Petrol == "1":
                Fuel_Type_Petrol = 1
                Fuel_Type_Diesel = 0
            else:
                Fuel_Type_Petrol = 0
                Fuel_Type_Diesel = 1
            Year = 2020 - Year
            Seller_Type_Individual = request.form["Seller_Type_Individual"]
            if Seller_Type_Individual == "2":
                Seller_Type_Individual = 1
            else:
                Seller_Type_Individual = 0
            Transmission_Mannual = request.form["Transmission_Mannual"]
            if Transmission_Mannual == "1":
                Transmission_Mannual = 1
            else:
                Transmission_Mannual = 0

            prediction = classifier2.predict(
                [
                    [
                        Present_Price,
                        Kms_Driven,
                        Owner,
                        Year,
                        Fuel_Type_Diesel,
                        Fuel_Type_Petrol,
                        Seller_Type_Individual,
                        Transmission_Mannual,
                    ]
                ]
            )

            return render_template("result.html", prediction_text=round(prediction[0]*100000, 2))

if __name__ == "__main__":
    app.run(debug=True)

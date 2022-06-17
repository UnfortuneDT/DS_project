import flask
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import render_template
import tensorflow as tf
import sklearn
import pickle
import keras
import numpy as np

app = flask.Flask(__name__, template_folder="templates")

model = keras.models.load_model('Neural model')

@app.route("/", methods = ["POST", "GET"])

@app.route("/index", methods = ["POST", "GET"])
def main():
    if flask.request.method == "GET":
        return render_template("main.html")


    if flask.request.method == "POST":
        
            first = float(flask.request.form["first_data"])
            second = float(flask.request.form["second_data"])
            third = float(flask.request.form["third_data"])
            fourth = float(flask.request.form["fourth_data"])
            fifth = float(flask.request.form["fifth_data"])
            sixth = float(flask.request.form["sixth_data"])
            seventh = float(flask.request.form["seventh_data"])
            eighth = float(flask.request.form["eighth_data"])
            ninth = float(flask.request.form["ninth_data"])
            eleventh = float(flask.request.form["eleventh_data"])
            twelfth = float(flask.request.form["twelfth_data"])
            thirtheenth = float(flask.request.form["thirtheenth_data"])
            test_array = pd.DataFrame([{"1":first,"2":second,"3":third,"4":fourth,"5":fifth,"6":sixth,"7":seventh,"8":eighth,"9":ninth,"10":eleventh,"11":twelfth,"12":thirtheenth}])
            minmax_scaler = MinMaxScaler()
            dataset_norm = minmax_scaler.fit_transform(np.array(test_array))
            X_bp_no_MinMax = pd.DataFrame(data=dataset_norm, columns=test_array.columns)
            y_pred = model.predict(X_bp_no_MinMax)

            return render_template("main.html", result = y_pred)

if __name__ == "__main__":
    app.run()



    
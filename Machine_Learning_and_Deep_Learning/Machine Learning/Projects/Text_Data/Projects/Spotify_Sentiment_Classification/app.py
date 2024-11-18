from flask import Flask, render_template, request
import joblib

app = Flask(__name__)



########################

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/prediction", methods = ['get', 'post'])
def prediction():
    review = request.form.get("review")
    
    model = joblib.load("best_models/naive_bayes.pkl")
    
    predicted = model.predict([review]) 
    
    if predicted[0] == 0:
        predicted = 'Negative'
    else:
        predicted = 'Positive'
       
    return render_template("prediction.html", predicted = predicted)


########################

if __name__ == '__main__':
    app.run(debug = True)
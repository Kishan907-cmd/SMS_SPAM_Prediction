
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predicts():

    message = request.form['message']

    message_tfidf = tfidf.transform([message])

    prediction = model.predict(message_tfidf)

    if prediction[0] == 1:
        result = "Spam Message"
    else:
        result = "Not Spam Message"

    return render_template('index.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import numpy as np
import pickle
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --- Get user inputs ---
        subscriberCount = float(request.form["subscriberCount"])
        viewCount = float(request.form["viewCount"])
        videoCount = float(request.form["videoCount"])
        avgLikes = float(request.form["avgLikes"])
        avgComments = float(request.form["avgComments"])
        creationDate = request.form["creationDate"]

        # --- Derived metrics ---
        # Convert creation date safely
        try:
            creation_date = datetime.strptime(creationDate, "%d-%m-%Y")
        except ValueError:
            creation_date = datetime.strptime(creationDate, "%Y-%m-%d")
        today = datetime.today()
        accountAgeDays = (today - creation_date).days

        # compute engagement rate and post frequency with safe guards
        if viewCount > 0:
            engagement_rate = round((avgLikes + avgComments) / viewCount, 5)
        else:
            engagement_rate = 0.0

        # Post frequency = videos per month
        if accountAgeDays > 0:
            post_frequency = videoCount / (accountAgeDays / 30.0)
        else:
            post_frequency = 0.0

        # --- Feature array (in same order as model was trained) ---
        features = np.array(
            [
                [
                    subscriberCount,
                    viewCount,
                    videoCount,
                    avgLikes,
                    avgComments,
                    engagement_rate,
                    post_frequency,
                    accountAgeDays,
                ]
            ]
        )

        # --- Scale features ---
        features_scaled = scaler.transform(features)

        # --- Predict popularity ---
        prediction = model.predict(features_scaled)[0]
        popularity_score = round(float(prediction), 2)

        # --- Determine rating style ---
        if popularity_score >= 8:
            emoji, color, rating = "🌟", "green", "Highly Popular"
        elif popularity_score >= 5:
            emoji, color, rating = "✨", "orange", "Moderately Popular"
        else:
            emoji, color, rating = "💤", "red", "Low Popularity"

        # --- Send all to template ---
        return render_template(
            "result.html",
            score=popularity_score,
            emoji=emoji,
            color=color,
            rating=rating,
            engagement_rate=round(engagement_rate, 6),
            post_frequency=round(post_frequency, 2),
            account_age=accountAgeDays,
        )

    except Exception as e:
        return render_template("result.html", score=None, error=str(e))


if __name__ == "__main__":
    app.run(debug=True)

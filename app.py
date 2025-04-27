from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# load sentiment-analysis pipeline once at startup
sentiment_analyzer = pipeline("sentiment-analysis")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        analysis = sentiment_analyzer(text)[0]
        result = {
            "label":   analysis["label"],
            "score":   f"{analysis['score']*100:.1f}%"
        }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

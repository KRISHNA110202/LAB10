from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

nltk.download('vader_lexicon')

@app.route('/')
def index():
    return render_template('index.html', result="", result_color="")

@app.route('/analyze', methods=['POST'])
def analyze():
    sid = SentimentIntensityAnalyzer()
    text = request.form.get('sentence')
    score = sid.polarity_scores(text)['compound']
    if score > 0:
        label = 'This sentence is positive'
        result_color = 'blue'  # Set color to blue for positive result
    elif score == 0:
        label = 'This sentence is neutral'
        result_color = 'orange'  # Set color to orange for neutral result
    else:
        label = 'This sentence is negative'
        result_color = 'red'  # Set color to red for negative result
    return render_template('index.html', result=label, result_color=result_color)

if __name__ == '__main__':
    app.run(debug=True)

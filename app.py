from flask import Flask, render_template, jsonify
import main as analysis
import os
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    # Run the analysis
    analysis.init_db().close()
    analysis.fetch_and_store_tweets()
    analysis.preprocess_data()
    analysis.analyze_sentiment()
    issues_df = analysis.analyze_main_issues()
    analysis.generate_insights()
    
    # Read the analysis results
    party_sentiment = pd.read_csv('outputs/party_sentiment.csv')
    issues_analysis = pd.read_csv('outputs/issues_analysis.csv')
    
    # Prepare data for frontend
    data = {
        'party_sentiment': party_sentiment.to_dict(orient='records'),
        'issues_analysis': issues_analysis.to_dict(orient='records'),
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return jsonify(data)

if __name__ == '__main__':
    # Create required directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Move output files to static folder
    output_files = [
        'sentiment_distribution.png',
        'party_sentiment_distribution.png',
        'issues_distribution.png',
        'issues_sentiment.png',
        'party_issue_heatmap.png',
        'wordcloud.png',
        'sentiment_trends.png'
    ]
    
    for file in output_files:
        src = f'outputs/{file}'
        dst = f'static/{file}'
        if os.path.exists(src):
            os.replace(src, dst)
    
    app.run(debug=True) 
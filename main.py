import pandas as pd
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
import os
from datetime import datetime, timedelta
import random
from wordcloud import WordCloud
import seaborn as sns

# --------------------------
# Configuration
# --------------------------
DATABASE = "data/election_analysis.db"

# --------------------------
# Create folders
# --------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --------------------------
# Sample Data Generation (for demonstration)
# --------------------------
SAMPLE_TWEETS = [
    # Education
    "AAP's education model has transformed Delhi's government schools #DelhiElections",
    "BJP promises more smart classrooms in Delhi schools #DelhiEducation",
    "Delhi's education budget shows AAP's commitment to learning",
    
    # Healthcare
    "Mohalla clinics by AAP have revolutionized healthcare in Delhi",
    "BJP pledges new super-specialty hospitals in Delhi #Healthcare",
    "Free healthcare services in Delhi need improvement",
    
    # Infrastructure
    "BJP's infrastructure projects have reduced traffic in Delhi",
    "Metro expansion under AAP government has improved connectivity",
    "New flyovers and roads development by BJP helping Delhi",
    
    # Pollution
    "Delhi's air quality remains a major concern under AAP",
    "BJP's central policies helping reduce Delhi pollution",
    "Need stronger measures to combat Delhi's pollution crisis",
    
    # Water Supply
    "24x7 water supply still a dream in many Delhi areas",
    "AAP's free water scheme has helped Delhi's poor",
    "BJP promises to solve Delhi's water crisis",
    
    # Safety
    "Women's safety in Delhi needs more attention",
    "CCTV installation by AAP government improving security",
    "BJP's law and order measures strengthening Delhi's safety",
    
    # Employment
    "Youth employment remains a challenge in Delhi",
    "AAP's job fairs creating opportunities for Delhi youth",
    "BJP's skill development programs helping Delhi's unemployment",
    
    # Additional mixed tweets for variety
    "Traffic situation in Delhi needs immediate attention",
    "Public transport system showing improvement under AAP",
    "BJP's development agenda looks promising for Delhi",
    "Healthcare system needs more funding in Delhi",
    "Education reforms showing positive results in Delhi"
]

def generate_sample_tweet():
    text = random.choice(SAMPLE_TWEETS)
    party = "AAP" if "AAP" in text else "BJP" if "BJP" in text else "Other"
    created_at = datetime.now() - timedelta(days=random.randint(0, 30))
    return text, created_at, party

# --------------------------
# Initialize database
# --------------------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text TEXT,
            cleaned_text TEXT,
            created_at DATETIME,
            sentiment_score REAL,
            sentiment_label TEXT,
            party TEXT
        )
    ''')
    conn.commit()
    return conn

# --------------------------
# Fetch and store sample tweets
# --------------------------
def fetch_and_store_tweets():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Generate and store 100 sample tweets
        for _ in range(100):
            text, created_at, party = generate_sample_tweet()
            cursor.execute('''
                INSERT INTO tweets (raw_text, created_at, party)
                VALUES (?, ?, ?)
            ''', (text, created_at, party))

        conn.commit()
        conn.close()
        print("100 Sample tweets stored successfully")

    except Exception as e:
        print(f"[Storage Error] {str(e)}")

# --------------------------
# Clean and preprocess data
# --------------------------
def preprocess_data():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql("SELECT * FROM tweets WHERE cleaned_text IS NULL", conn)

    if df.empty:
        print("No new tweets to preprocess.")
        conn.close()
        return

    df['cleaned_text'] = df['raw_text'].apply(lambda x: re.sub(
        r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", "", x).lower().strip())

    for _, row in df.iterrows():
        conn.execute('''
            UPDATE tweets 
            SET cleaned_text = ?
            WHERE id = ?
        ''', (row['cleaned_text'], row['id']))

    conn.commit()
    conn.close()
    print("Preprocessing completed")

# --------------------------
# Sentiment Analysis
# --------------------------
def analyze_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql("SELECT * FROM tweets WHERE sentiment_score IS NULL", conn)

    if df.empty:
        print("No tweets found for sentiment analysis.")
        conn.close()
        return

    df['sentiment_score'] = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: "Positive" if x > 0.05 else
        "Negative" if x < -0.05 else "Neutral"
    )

    for _, row in df.iterrows():
        conn.execute('''
            UPDATE tweets 
            SET sentiment_score = ?, sentiment_label = ?
            WHERE id = ?
        ''', (row['sentiment_score'], row['sentiment_label'], row['id']))

    conn.commit()
    conn.close()
    print("Sentiment analysis completed")

# --------------------------
# Generate Insights & Plots
# --------------------------
def generate_insights():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql("SELECT * FROM tweets WHERE cleaned_text IS NOT NULL", conn)

    if df.empty:
        print("No data available to generate insights.")
        conn.close()
        return

    # Party-wise sentiment analysis
    party_sentiment = df.groupby('party')['sentiment_score'].agg(['mean', 'count']).round(3)
    print("\nParty-wise Sentiment Analysis:")
    print(party_sentiment)
    
    # Save party sentiment to CSV
    party_sentiment.to_csv('outputs/party_sentiment.csv')

    # Sentiment Distribution Plot
    plt.figure(figsize=(10, 6))
    df['sentiment_label'].value_counts().plot(kind='bar')
    plt.title('Overall Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.tight_layout()
    plt.savefig('outputs/sentiment_distribution.png')
    plt.close()

    # Party-wise Sentiment Distribution
    plt.figure(figsize=(12, 6))
    sentiment_by_party = pd.crosstab(df['party'], df['sentiment_label'])
    sentiment_by_party.plot(kind='bar', stacked=True)
    plt.title('Party-wise Sentiment Distribution')
    plt.xlabel('Political Party')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig('outputs/party_sentiment_distribution.png')
    plt.close()

    # Time series analysis
    df['created_at'] = pd.to_datetime(df['created_at'])
    daily_sentiment = df.groupby([df['created_at'].dt.date, 'party'])['sentiment_score'].mean().unstack()
    
    plt.figure(figsize=(12, 6))
    daily_sentiment.plot(marker='o')
    plt.title('Sentiment Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.legend(title='Political Party')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/sentiment_trends.png')
    plt.close()

    # Save detailed analysis to CSV
    df.to_csv('outputs/detailed_analysis.csv', index=False)
    
    print("\nAnalysis completed! Check the 'outputs' folder for:")
    print("1. sentiment_distribution.png - Overall sentiment distribution")
    print("2. party_sentiment_distribution.png - Party-wise sentiment breakdown")
    print("3. sentiment_trends.png - Sentiment trends over time")
    print("4. party_sentiment.csv - Detailed party-wise sentiment metrics")
    print("5. detailed_analysis.csv - Complete analysis data")

    conn.close()

def analyze_main_issues():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql("SELECT * FROM tweets", conn)
    
    # Define main issues and their keywords
    issues = {
        'Education': ['education', 'school', 'learning', 'student', 'classroom'],
        'Healthcare': ['health', 'hospital', 'clinic', 'medical', 'doctor'],
        'Infrastructure': ['infrastructure', 'road', 'metro', 'transport', 'traffic'],
        'Pollution': ['pollution', 'air quality', 'environment', 'clean'],
        'Water': ['water', 'supply', 'pipeline'],
        'Safety': ['safety', 'security', 'crime', 'police', 'cctv'],
        'Employment': ['job', 'employment', 'career', 'work', 'salary']
    }
    
    # Analyze issues in tweets
    issue_counts = {issue: 0 for issue in issues}
    issue_sentiment = {issue: [] for issue in issues}
    
    for _, tweet in df.iterrows():
        text = tweet['raw_text'].lower()
        for issue, keywords in issues.items():
            if any(keyword in text for keyword in keywords):
                issue_counts[issue] += 1
                issue_sentiment[issue].append(tweet['sentiment_score'])
    
    # Calculate average sentiment for each issue
    issue_analysis = {
        issue: {
            'count': count,
            'avg_sentiment': sum(issue_sentiment[issue])/len(issue_sentiment[issue]) if issue_sentiment[issue] else 0
        }
        for issue, count in issue_counts.items()
    }
    
    # Create issues DataFrame
    issues_df = pd.DataFrame.from_dict(issue_analysis, orient='index')
    issues_df.to_csv('outputs/issues_analysis.csv')
    
    # Plot issue distribution
    plt.figure(figsize=(12, 6))
    plt.bar(issue_counts.keys(), issue_counts.values())
    plt.title('Main Issues Distribution in Delhi Elections')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/issues_distribution.png')
    plt.close()
    
    # Plot issue sentiment
    plt.figure(figsize=(12, 6))
    sentiment_data = [(issue, data['avg_sentiment']) for issue, data in issue_analysis.items()]
    sentiment_df = pd.DataFrame(sentiment_data, columns=['Issue', 'Sentiment'])
    sns.barplot(x='Issue', y='Sentiment', data=sentiment_df)
    plt.title('Sentiment Analysis by Issue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/issues_sentiment.png')
    plt.close()
    
    # Generate word cloud
    text = ' '.join(df['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Discussed Topics')
    plt.tight_layout()
    plt.savefig('outputs/wordcloud.png')
    plt.close()
    
    # Create party-wise issue analysis
    party_issue_sentiment = {}
    for party in df['party'].unique():
        party_tweets = df[df['party'] == party]
        party_issue_sentiment[party] = {}
        for issue, keywords in issues.items():
            relevant_tweets = party_tweets[party_tweets['raw_text'].str.lower().apply(
                lambda x: any(keyword in x for keyword in keywords))]
            if not relevant_tweets.empty:
                party_issue_sentiment[party][issue] = relevant_tweets['sentiment_score'].mean()
    
    # Plot party-wise issue sentiment
    party_issue_df = pd.DataFrame(party_issue_sentiment)
    plt.figure(figsize=(12, 8))
    sns.heatmap(party_issue_df, annot=True, cmap='RdYlGn', center=0)
    plt.title('Party-wise Sentiment on Different Issues')
    plt.tight_layout()
    plt.savefig('outputs/party_issue_heatmap.png')
    plt.close()
    
    return issues_df

def generate_presentation_html():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Delhi Elections Sentiment Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .section { margin-bottom: 40px; }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; }
            img { max-width: 100%; margin: 20px 0; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .highlight { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Delhi Elections 2025: Sentiment Analysis Report</h1>
        
        <div class="section">
            <h2>1. Overall Sentiment Distribution</h2>
            <img src="sentiment_distribution.png" alt="Sentiment Distribution">
            <p class="highlight">This graph shows the distribution of positive, negative, and neutral sentiments across all analyzed tweets.</p>
        </div>
        
        <div class="section">
            <h2>2. Party-wise Sentiment Analysis</h2>
            <img src="party_sentiment_distribution.png" alt="Party Sentiment Distribution">
            <p class="highlight">Comparison of sentiment distribution between different political parties.</p>
        </div>
        
        <div class="section">
            <h2>3. Main Issues in Delhi Elections</h2>
            <img src="issues_distribution.png" alt="Issues Distribution">
            <p class="highlight">Analysis of the main issues being discussed in the context of Delhi elections.</p>
        </div>
        
        <div class="section">
            <h2>4. Issue-wise Sentiment Analysis</h2>
            <img src="issues_sentiment.png" alt="Issues Sentiment">
            <p class="highlight">Shows how different issues are perceived by the public (positive or negative sentiment).</p>
        </div>
        
        <div class="section">
            <h2>5. Party Stance on Issues</h2>
            <img src="party_issue_heatmap.png" alt="Party Issue Heatmap">
            <p class="highlight">Heatmap showing how different parties are perceived on various issues.</p>
        </div>
        
        <div class="section">
            <h2>6. Key Topics Word Cloud</h2>
            <img src="wordcloud.png" alt="Word Cloud">
            <p class="highlight">Visual representation of the most frequently discussed topics.</p>
        </div>
        
        <div class="section">
            <h2>7. Sentiment Trends Over Time</h2>
            <img src="sentiment_trends.png" alt="Sentiment Trends">
            <p class="highlight">Shows how sentiments have evolved over the analysis period.</p>
        </div>
    </body>
    </html>
    """
    
    with open('outputs/presentation.html', 'w') as f:
        f.write(html_content)

# --------------------------
# Run all steps
# --------------------------
if __name__ == "__main__":
    print("Starting Delhi Elections Sentiment Analysis...")
    init_db().close()
    fetch_and_store_tweets()
    preprocess_data()
    analyze_sentiment()
    analyze_main_issues()
    generate_insights()
    generate_presentation_html()
    print("\nAnalysis completed! Check the 'outputs' folder for:")
    print("1. presentation.html - Complete analysis presentation")
    print("2. Various visualization files")
    print("3. Detailed CSV reports")

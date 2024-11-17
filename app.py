import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import shap
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
except ImportError:
    raise ImportError("The SHAP library is not installed. Please install it using 'pip install shap'")

# Streamlit Configuration
st.set_page_config(page_title="SWOT Leadership Analysis", page_icon="🌟", layout="wide")

# Define Watermark
WATERMARK = "Advanced AI Leadership Analysis by Muhammad Allam Rafi, CBOA® CDSP®"

# Load Transformer Model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Define Leadership Traits for SWOT
LEADERSHIP_TRAITS = {
    "Positive": {
        "Leadership": "Ability to lead and inspire others.",
        "Vision": "Clear and inspiring direction for the future.",
        "Integrity": "Acting consistently with honesty and strong ethics.",
        "Innovation": "Driving creativity and fostering change.",
        "Inclusivity": "Promoting diversity and creating an inclusive environment.",
        "Empathy": "Understanding others' perspectives and feelings.",
        "Communication": "Conveying ideas clearly and effectively.",
    },
    "Neutral": {
        "Adaptability": "Flexibility to adjust to new challenges.",
        "Time Management": "Prioritizing and organizing tasks efficiently.",
        "Problem-Solving": "Resolving issues effectively.",
        "Conflict Resolution": "Managing disagreements constructively.",
        "Resilience": "Bouncing back from setbacks.",
    },
    "Negative": {
        "Micromanagement": "Excessive control over tasks.",
        "Overconfidence": "Ignoring input due to arrogance.",
        "Conflict Avoidance": "Avoiding necessary confrontations.",
        "Indecisiveness": "Inability to make timely decisions.",
        "Rigidity": "Refusing to adapt to new circumstances.",
    }
}

CATEGORY_WEIGHTS = {"Strengths": 1.5, "Weaknesses": 1.3, "Opportunities": 1.4, "Threats": 1.2}

# Sidebar Profile and Contact
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="Muhammad Allam Rafi", use_column_width=True)
    st.markdown("### **🌟 About Me 🌟**")
    st.markdown("""
    👨‍⚕️ **Medical Student**  
    Passionate about **Machine Learning**, **Leadership Research**, and **Healthcare AI**.  
    - 🎓 **Faculty of Medicine**, Universitas Indonesia  
    - 📊 **Research Interests**:  
      - Leadership Viability in Healthcare  
      - AI-driven solutions for medical challenges  
      - Natural Language Processing and Behavioral Analysis  
    - 🧑‍💻 **Skills**: Python, NLP, Data Visualization
    """)
    st.markdown("### **📫 Contact Me**")
    st.markdown("""
    - [LinkedIn](https://linkedin.com)  
    - [GitHub](https://github.com)  
    - [Email](mailto:allamrafi@example.com)  
    """)
    st.markdown(f"---\n🌟 **{WATERMARK}** 🌟")

# Header Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌟 Advanced SWOT Leadership Analysis 🌟</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #808080;'>Discover Leadership Potential with Explainable AI</h3>", unsafe_allow_html=True)
st.markdown("---")

# NLP Analysis Function
def analyze_text_with_shap(text, traits, confidence, category_weight):
    """Analyze text using SHAP-like explanations."""
    if not text.strip():
        return {}, {}

    scores, explanations = {}, {}
    text_embedding = model.encode(text, convert_to_tensor=True)
    trait_embeddings = model.encode(list(traits.values()), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(text_embedding, trait_embeddings).squeeze().tolist()

    for trait, similarity in zip(traits.keys(), similarities):
        weighted_score = similarity * (confidence / 10) * category_weight
        scores[trait] = max(0, weighted_score)  # Ensure scores are non-negative
        explanations[trait] = f"'{text}' aligns with '{trait}'. Similarity: {similarity:.2f}, Weighted Score: {weighted_score:.2f}."

    return scores, explanations

# Input Validation
def validate_swot_inputs(swot_inputs):
    for category, entries in swot_inputs.items():
        for text, _ in entries:
            if text.strip():
                return True
    return False

def validate_behavioral_inputs(behavior_inputs):
    for response in behavior_inputs.values():
        if response.strip():
            return True
    return False

# Input Fields
swot_inputs = {
    category: [
        (st.text_area(f"{category} #{i+1}", key=f"{category}_{i}"), 
         st.slider(f"{category} #{i+1} Confidence", 1, 10, 5, key=f"{category}_confidence_{i}"))
        for i in range(3)
    ]
    for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]
}

behavior_questions = {
    "Q1": "Describe how you handle stress.",
    "Q2": "What motivates you to lead others?",
    "Q3": "How do you approach conflict resolution?",
    "Q4": "What is your strategy for long-term planning?",
    "Q5": "How do you inspire teamwork in challenging situations?"
}
behavior_responses = {q: st.text_area(q, key=f"behavior_{i}") for i, q in enumerate(behavior_questions.values())}

# 3D Scatter Plot
def generate_3d_scatter(data):
    """Generates 3D scatter plot."""
    fig = go.Figure()
    categories = list(data.keys())
    for idx, (category, traits) in enumerate(data.items()):
        xs = list(range(len(traits)))
        ys = [idx] * len(traits)
        zs = list(traits.values())

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(size=5, color=zs, colorscale='Viridis'),
            name=category
        ))

    fig.update_layout(
        title="3D Scatter Plot - SWOT Analysis",
        scene=dict(xaxis_title="Traits", yaxis_title="Categories", zaxis_title="Scores")
    )
    return fig

# Advanced Behavioral Analysis (Meta-Learning Inspired)
def behavioral_meta_learning(behavior_responses, swot_scores):
    """
    Adapt behavioral scoring dynamically based on patterns from SWOT scores.
    """
    behavior_scores = {}
    for question, response in behavior_responses.items():
        if response.strip():
            # Heuristic: Use average of Strengths and Opportunities to adapt behavioral scoring
            strengths_avg = np.mean(list(swot_scores.get("Strengths", {}).values()) or [0])
            opportunities_avg = np.mean(list(swot_scores.get("Opportunities", {}).values()) or [0])
            behavior_scores[question] = (strengths_avg + opportunities_avg) * 0.7  # Adjusted weight
    return behavior_scores

# Advanced Machine Learning: KMeans Clustering for SWOT
def kmeans_clustering(data):
    """
    Perform clustering on SWOT traits for enhanced visualization.
    """
    all_traits = []
    trait_labels = []
    for category, traits in data.items():
        all_traits.extend(list(traits.values()))
        trait_labels.extend([f"{category}-{trait}" for trait in traits.keys()])

    # Reshape for clustering
    traits_array = np.array(all_traits).reshape(-1, 1)

    # Perform KMeans clustering
    n_clusters = min(len(traits_array), 3)  # Limit clusters to avoid over-segmentation
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(traits_array)

    # Create dataframe for visualization
    cluster_df = pd.DataFrame({
        "Trait": trait_labels,
        "Score": all_traits,
        "Cluster": kmeans.labels_
    })
    return cluster_df

# Advanced Visualization: SHAP for Explainability
def visualize_shap_values(traits, shap_values):
    """
    Generate a bar chart for SHAP values to explain trait alignment.
    """
    st.markdown("### SHAP Explanation: Alignment of Input with Traits")
    fig = go.Figure(go.Bar(
        x=shap_values,
        y=traits,
        orientation='h',
        marker=dict(color='blue'),
    ))
    fig.update_layout(
        title="Explainable Alignment with Leadership Traits",
        xaxis_title="SHAP Value",
        yaxis_title="Traits"
    )
    st.plotly_chart(fig, use_container_width=True)

# Advanced Visualization: Interactive 3D Surface
def generate_3d_surface(data):
    """
    Generates an interactive 3D surface plot using Plotly.
    """
    categories = list(data.keys())
    traits = list(next(iter(data.values())).keys())
    z = np.array([list(traits.values()) for traits in data.values()])
    x, y = np.meshgrid(range(len(categories)), range(len(traits)))

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title="3D Surface Plot: SWOT Scores",
        scene=dict(
            xaxis=dict(title="Categories"),
            yaxis=dict(title="Traits"),
            zaxis=dict(title="Scores"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# Generate Advanced PDF Report
class AdvancedPDF(FPDF):
    """
    Custom PDF class for generating professional SWOT Leadership reports.
    """
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "Advanced SWOT Leadership Analysis Report", align='C', ln=True)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {WATERMARK}", align='C')

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, content)

def generate_pdf_report(swot_scores, lsi, lsi_interpretation, behavior_scores, cluster_df, chart_paths):
    """
    Generate PDF with SWOT analysis results, behavioral analysis, and visualizations.
    """
    pdf = AdvancedPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Leadership Viability Index (LSI): {lsi:.2f}", ln=True)
    pdf.cell(0, 10, f"Interpretation: {lsi_interpretation}", ln=True)

    # Add Behavioral Analysis
    pdf.add_section("Behavioral Analysis", "\n".join([f"{q}: {s:.2f}" for q, s in behavior_scores.items()]))

    # Add SWOT Scores and Clusters
    for category, traits in swot_scores.items():
        pdf.add_section(f"{category} Scores", "\n".join([f"{trait}: {value:.2f}" for trait, value in traits.items()]))
    pdf.add_section("KMeans Clustering", cluster_df.to_string(index=False))

    # Add Charts
    for chart_path in chart_paths:
        pdf.add_page()
        pdf.image(chart_path, x=10, y=50, w=190)
    pdf.output("/tmp/advanced_report.pdf")
    return "/tmp/advanced_report.pdf"

# Execution: Perform Full Analysis
if st.button("Analyze"):
    if not (validate_swot_inputs(swot_inputs) or validate_behavioral_inputs(behavior_responses)):
        st.error("Please provide at least one valid SWOT input or Behavioral response.")
    else:
        st.success("Analysis in progress...")

        # Process SWOT Inputs
        swot_scores = {}
        for category, entries in swot_inputs.items():
            qualities = (
                LEADERSHIP_TRAITS["Positive"] if category in ["Strengths", "Opportunities"] else
                LEADERSHIP_TRAITS["Negative"] if category == "Threats" else
                LEADERSHIP_TRAITS["Neutral"]
            )
            category_scores = {}
            for text, confidence in entries:
                if text.strip():
                    scores, _ = analyze_text_with_shap(text, qualities, confidence, CATEGORY_WEIGHTS[category])
                    category_scores.update(scores)
            swot_scores[category] = category_scores or {trait: 0 for trait in qualities.keys()}

        # Behavioral Analysis
        behavior_scores = behavioral_meta_learning(behavior_responses, swot_scores)

        # LSI Calculation
        total_strengths = sum(swot_scores.get("Strengths", {}).values())
        total_weaknesses = sum(swot_scores.get("Weaknesses", {}).values())
        lsi = np.log((total_strengths + 1) / (total_weaknesses + 1))
        lsi_interpretation = (
            "Exceptional Leadership Potential" if lsi > 1.5 else
            "Good Leadership Potential" if lsi > 0.5 else
            "Moderate Leadership Potential" if lsi > -0.5 else
            "Needs Improvement"
        )

        # Display Results
        st.subheader(f"Leadership Viability Index (LSI): {lsi:.2f}")
        st.write(f"**Interpretation**: {lsi_interpretation}")

        # Generate Visualizations
        st.markdown("### SWOT 2D and 3D Visualizations")
        generate_3d_surface(swot_scores)

        # KMeans Clustering
        cluster_df = kmeans_clustering(swot_scores)
        st.write("### SWOT Clustering Results")
        st.dataframe(cluster_df)

        # Generate PDF
        st.markdown("### Generate Report")
        pdf_path = generate_pdf_report(swot_scores, lsi, lsi_interpretation, behavior_scores, cluster_df, [])
        with open(pdf_path, "rb") as pdf_file:
            st.download_button("Download Full Report", pdf_file, "SWOT_Report.pdf", mime="application/pdf")

import torch
import torch.nn as nn
from transformers import pipeline

# Define a Deep Learning Model for Regression
class LeadershipPotentialModel(nn.Module):
    def __init__(self, input_dim):
        super(LeadershipPotentialModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Train Predictive Model
def train_regression_model(data, targets):
    """
    Train a simple regression model to predict Leadership Potential Score.
    """
    input_dim = data.shape[1]
    model = LeadershipPotentialModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    # Training loop
    for epoch in range(200):  # Iterasi 200 epoch
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs, targets_tensor)
        loss.backward()
        optimizer.step()

    return model

# Generate Auto-Recommendations
def generate_recommendations(swot_scores):
    """
    Generate actionable recommendations based on SWOT scores.
    """
    recommendations = []
    for category, traits in swot_scores.items():
        for trait, score in traits.items():
            if category == "Strengths" and score > 0.7:
                recommendations.append(f"Leverage your strength in '{trait}' to inspire others.")
            elif category == "Weaknesses" and score > 0.5:
                recommendations.append(f"Work on improving your '{trait}' to minimize its impact.")
            elif category == "Opportunities" and score > 0.6:
                recommendations.append(f"Explore opportunities related to '{trait}' to grow.")
            elif category == "Threats" and score > 0.4:
                recommendations.append(f"Mitigate threats in '{trait}' to ensure sustainability.")
    return recommendations

# Advanced Sentiment Analysis
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

def analyze_sentiment(swot_inputs):
    """
    Perform sentiment analysis on SWOT inputs.
    """
    sentiments = {}
    for category, entries in swot_inputs.items():
        sentiments[category] = []
        for text, _ in entries:
            if text.strip():
                sentiment_result = sentiment_model(text[:512])  # Limit text length to 512 chars
                sentiments[category].append(sentiment_result[0])
            else:
                sentiments[category].append({"label": "Neutral", "score": 0.0})
    return sentiments

# 3D Time-Series Visualization
def generate_3d_time_series(data, time_range):
    """
    Generate a 3D time-series plot showing the evolution of SWOT scores.
    """
    fig = go.Figure()

    categories = list(data.keys())
    time_steps = range(len(time_range))
    for i, category in enumerate(categories):
        for j, (trait, scores) in enumerate(data[category].items()):
            fig.add_trace(go.Scatter3d(
                x=time_steps, 
                y=[i] * len(time_steps), 
                z=scores,
                mode='lines',
                name=f"{category} - {trait}"
            ))

    fig.update_layout(
        title="3D Time-Series SWOT Visualization",
        scene=dict(
            xaxis=dict(title="Time"),
            yaxis=dict(title="Categories"),
            zaxis=dict(title="Scores")
        )
    )
    return fig

# Behavioral Clustering Analysis
def cluster_behavioral_responses(behavior_responses):
    """
    Cluster behavioral responses using KMeans for pattern detection.
    """
    embeddings = []
    for response in behavior_responses.values():
        if response.strip():
            embedding = model.encode(response, convert_to_tensor=False)
            embeddings.append(embedding)

    if len(embeddings) > 1:
        kmeans = KMeans(n_clusters=min(3, len(embeddings)), random_state=42)
        labels = kmeans.fit_predict(embeddings)
        clustered_responses = {f"Cluster {label}": [] for label in set(labels)}
        for idx, label in enumerate(labels):
            clustered_responses[f"Cluster {label}"].append(list(behavior_responses.values())[idx])
    else:
        clustered_responses = {"Cluster 0": list(behavior_responses.values())}

    return clustered_responses

# Multilingual NLP Integration
def multilingual_analysis(text, language):
    """
    Perform analysis in different languages using Hugging Face transformers.
    """
    translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-{lang}".format(lang=language))
    translated_text = translation_pipeline(text)[0]['translation_text']
    embedding = model.encode(translated_text, convert_to_tensor=True)
    return embedding

# Execution: Perform Full Advanced Analysis
if st.button("Advanced Analyze"):
    if not (validate_swot_inputs(swot_inputs) or validate_behavioral_inputs(behavior_responses)):
        st.error("Please provide at least one valid SWOT input or Behavioral response.")
    else:
        st.success("Advanced Analysis in progress...")

        # Sentiment Analysis
        st.markdown("### Sentiment Analysis")
        sentiments = analyze_sentiment(swot_inputs)
        st.json(sentiments)

        # Clustering
        st.markdown("### Behavioral Clustering")
        clusters = cluster_behavioral_responses(behavior_responses)
        st.json(clusters)

        # Auto-Recommendations
        recommendations = generate_recommendations(swot_scores)
        st.markdown("### AI Recommendations")
        for rec in recommendations:
            st.markdown(f"- {rec}")

        # Advanced 3D Time-Series
        time_series_data = {cat: {trait: np.random.rand(5).tolist() for trait in traits.keys()} for cat, traits in swot_scores.items()}
        fig = generate_3d_time_series(time_series_data, time_range=["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"])
        st.plotly_chart(fig, use_container_width=True)

from keras.models import Sequential
from keras.layers import LSTM, Dense
from pyvis.network import Network

# Time-Series Forecasting with LSTM
def train_lstm_model(data):
    """
    Train an LSTM model to predict future SWOT scores.
    """
    data = np.array(data).reshape(-1, 1)
    X, y = [], []
    for i in range(len(data) - 3):
        X.append(data[i:i+3])
        y.append(data[i+3])
    X, y = np.array(X), np.array(y)

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)

    return model

def predict_future_scores(model, data, steps=5):
    """
    Predict future SWOT scores using trained LSTM model.
    """
    predictions = []
    input_seq = np.array(data[-3:]).reshape(1, 3, 1)
    for _ in range(steps):
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)
    return predictions

# Real-Time Feedback on SWOT Input
def real_time_feedback(input_text, category):
    """
    Provide real-time feedback based on SWOT input.
    """
    if category == "Strengths":
        if "team" in input_text.lower():
            return "Excellent! Highlighting team-building is a strong leadership quality."
        elif "vision" in input_text.lower():
            return "Great! Visionary thinking is a critical leadership trait."
    elif category == "Weaknesses":
        if "time management" in input_text.lower():
            return "Consider strategies to improve time management for better results."
        elif "indecisiveness" in input_text.lower():
            return "Try focusing on decision-making frameworks to overcome indecisiveness."
    return "Keep refining your input for more actionable insights."

# Leadership Growth Path
def create_growth_path(swot_scores, behavior_scores):
    """
    Generate a leadership growth path based on analysis.
    """
    growth_path = []
    strengths = swot_scores.get("Strengths", {})
    weaknesses = swot_scores.get("Weaknesses", {})
    opportunities = swot_scores.get("Opportunities", {})

    if strengths:
        top_strength = max(strengths, key=strengths.get)
        growth_path.append(f"Leverage your strength in '{top_strength}' to inspire others.")
    if weaknesses:
        top_weakness = max(weaknesses, key=weaknesses.get)
        growth_path.append(f"Work on improving your '{top_weakness}' to enhance leadership capabilities.")
    if opportunities:
        top_opportunity = max(opportunities, key=opportunities.get)
        growth_path.append(f"Explore opportunities in '{top_opportunity}' for professional growth.")

    return growth_path

# SWOT Relationship Mapping
def create_relationship_map(swot_scores):
    """
    Create a relationship map of SWOT categories and traits using PyVis.
    """
    net = Network(height="500px", width="100%", notebook=True)
    net.add_node("SWOT", color="blue", size=30)

    for category, traits in swot_scores.items():
        net.add_node(category, color="green", size=20)
        net.add_edge("SWOT", category)
        for trait, score in traits.items():
            net.add_node(trait, title=f"Score: {score:.2f}", size=10)
            net.add_edge(category, trait)

    net.show("relationship_map.html")

# Risk Assessment
def assess_risks(threats):
    """
    Perform risk assessment and suggest mitigation strategies.
    """
    risk_report = []
    for threat, score in threats.items():
        if score > 0.7:
            risk_report.append(f"High risk identified in '{threat}'. Consider immediate mitigation.")
        elif score > 0.5:
            risk_report.append(f"Moderate risk in '{threat}'. Monitor and plan mitigation strategies.")
    return risk_report

# Gamification: Progress Tracker
def track_progress(completed_tasks, total_tasks):
    """
    Add gamification element to track progress.
    """
    progress = (completed_tasks / total_tasks) * 100
    st.markdown(f"### 🎯 Progress: {progress:.2f}%")
    st.progress(progress / 100)

# Main Execution: Premium Features
if st.button("Premium Analyze"):
    if not (validate_swot_inputs(swot_inputs) or validate_behavioral_inputs(behavior_responses)):
        st.error("Please provide at least one valid SWOT input or Behavioral response.")
    else:
        st.success("Performing Premium Analysis...")

        # LSTM Forecasting
        st.markdown("### SWOT Forecasting")
        forecast_data = [0.3, 0.5, 0.7, 0.6, 0.8]  # Example data
        lstm_model = train_lstm_model(forecast_data)
        future_scores = predict_future_scores(lstm_model, forecast_data)
        st.line_chart(future_scores)

        # Feedback Loop
        st.markdown("### Real-Time Feedback")
        for category, entries in swot_inputs.items():
            for text, _ in entries:
                feedback = real_time_feedback(text, category)
                st.write(f"Feedback for {category}: {feedback}")

        # Leadership Growth Path
        st.markdown("### Leadership Growth Path")
        growth_path = create_growth_path(swot_scores, behavior_scores)
        for step in growth_path:
            st.markdown(f"- {step}")

        # Relationship Mapping
        st.markdown("### SWOT Relationship Mapping")
        create_relationship_map(swot_scores)
        st.markdown("View the generated **relationship map** [here](relationship_map.html).")

        # Risk Assessment
        st.markdown("### Risk Assessment")
        risks = assess_risks(swot_scores.get("Threats", {}))
        for risk in risks:
            st.markdown(f"- {risk}")

        # Gamification
        st.markdown("### Gamification Progress Tracker")
        track_progress(completed_tasks=4, total_tasks=5)

import openai
import networkx as nx
from ortools.linear_solver import pywraplp

# OpenAI GPT API Key (Replace with your API key)
openai.api_key = "your-api-key-here"

# Dynamic Leadership Simulation (DLS)
def simulate_leadership_scenarios(swot_scores):
    """
    Simulate dynamic scenarios based on SWOT analysis.
    """
    scenarios = [
        "What happens if your top strength is removed?",
        "How can you overcome your top weakness?",
        "What opportunities align best with your strengths?",
        "How can threats be minimized with available strengths?"
    ]
    insights = []
    for scenario in scenarios:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Based on the SWOT scores {swot_scores}, analyze the following scenario: {scenario}",
            max_tokens=150
        )
        insights.append(response["choices"][0]["text"].strip())
    return insights

# Deep Sentiment Calibration
def calibrate_sentiment(sentiments):
    """
    Map sentiment analysis to a granular scale of 1-100.
    """
    calibrated_scores = {}
    for category, sentiment_list in sentiments.items():
        calibrated_scores[category] = []
        for sentiment in sentiment_list:
            if sentiment["label"] == "POSITIVE":
                calibrated_scores[category].append(sentiment["score"] * 100)
            elif sentiment["label"] == "NEGATIVE":
                calibrated_scores[category].append(sentiment["score"] * -100)
            else:
                calibrated_scores[category].append(0)
    return calibrated_scores

# Neural Narrative Generator (NNG)
def generate_swot_report(swot_scores, recommendations, growth_path):
    """
    Generate a professional SWOT report using OpenAI GPT.
    """
    prompt = f"""
    Generate a professional narrative report based on the following SWOT analysis:
    SWOT Scores: {swot_scores}
    Recommendations: {recommendations}
    Leadership Growth Path: {growth_path}
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response["choices"][0]["text"].strip()

# Predictive SWOT Optimization
def optimize_swot(swot_scores):
    """
    Optimize SWOT scores using a linear programming model.
    """
    solver = pywraplp.Solver.CreateSolver('GLOP')
    variables = {}
    objective = solver.Objective()

    for category, traits in swot_scores.items():
        for trait, score in traits.items():
            var = solver.NumVar(0, 1, f"{category}_{trait}")
            variables[f"{category}_{trait}"] = var
            objective.SetCoefficient(var, score)

    objective.SetMaximization()
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        optimized_scores = {var.name(): var.solution_value() for var in variables.values()}
        return optimized_scores
    else:
        return "Optimization failed"

# Multi-User Collaboration Dashboard
def collaboration_dashboard(swot_scores, user_inputs):
    """
    Create a collaboration dashboard for team analysis.
    """
    G = nx.Graph()
    G.add_node("SWOT Analysis", color="blue", size=30)

    for user, scores in user_inputs.items():
        G.add_node(user, color="green", size=20)
        G.add_edge("SWOT Analysis", user)

        for category, traits in scores.items():
            for trait, score in traits.items():
                trait_node = f"{user}_{category}_{trait}"
                G.add_node(trait_node, title=f"Score: {score:.2f}", size=10)
                G.add_edge(user, trait_node)

    net = Network(height="500px", width="100%", notebook=True)
    net.from_nx(G)
    net.show("collaboration_dashboard.html")

# Execution: Advanced Premium Features
if st.button("Final Analysis"):
    if not (validate_swot_inputs(swot_inputs) or validate_behavioral_inputs(behavior_responses)):
        st.error("Please provide at least one valid SWOT input or Behavioral response.")
    else:
        st.success("Performing the final analysis...")

        # Leadership Simulation
        st.markdown("### Leadership Simulation Insights")
        simulation_insights = simulate_leadership_scenarios(swot_scores)
        for insight in simulation_insights:
            st.markdown(f"- {insight}")

        # Sentiment Calibration
        st.markdown("### Calibrated Sentiment Scores")
        calibrated_sentiments = calibrate_sentiment(sentiments)
        st.json(calibrated_sentiments)

        # Neural Narrative Report
        st.markdown("### Professional SWOT Report")
        report = generate_swot_report(swot_scores, recommendations, growth_path)
        st.write(report)

        # SWOT Optimization
        st.markdown("### Optimized SWOT Scores")
        optimized_scores = optimize_swot(swot_scores)
        st.json(optimized_scores)

        # Collaboration Dashboard
        st.markdown("### Multi-User Collaboration Dashboard")
        collaboration_dashboard(swot_scores, {"User 1": swot_scores, "User 2": swot_scores})
        st.markdown("View the generated **collaboration dashboard** [here](collaboration_dashboard.html).")

from fpdf import FPDF
from datetime import datetime

class ProfessionalPDF(FPDF):
    """
    Generate a professional PDF report for SWOT Leadership Analysis.
    """

    def header(self):
        # Set header font
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "Professional SWOT Leadership Analysis Report", align='C', ln=True)
        self.ln(10)  # Add a line break

    def footer(self):
        # Set footer with date and watermark
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Advanced AI Leadership Tools", align='C')

    def add_section(self, title, content):
        # Add section with title and content
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)  # Line break
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, content)
        self.ln(5)  # Line break

    def add_table(self, data, column_names):
        # Add a table to the PDF
        self.set_font('Arial', 'B', 10)
        col_width = self.w / (len(column_names) + 1)  # Calculate column width
        for col in column_names:
            self.cell(col_width, 10, col, border=1, align='C')
        self.ln()
        self.set_font('Arial', '', 10)
        for row in data:
            for item in row:
                self.cell(col_width, 10, str(item), border=1, align='C')
            self.ln()

    def add_image(self, img_path, x=10, y=None, w=190):
        # Add an image to the PDF
        self.image(img_path, x=x, y=y, w=w)
        self.ln(10)

def generate_professional_pdf(swot_scores, sentiments, recommendations, growth_path, chart_paths):
    """
    Generate a comprehensive professional PDF report for SWOT Analysis.
    """
    pdf = ProfessionalPDF()
    pdf.add_page()

    # Executive Summary
    pdf.add_section("Executive Summary", """
    This report provides an in-depth analysis of your leadership potential based on SWOT and Behavioral analysis.
    Key insights, recommendations, and growth pathways are detailed in the following sections.
    """)

    # SWOT Analysis Scores
    for category, traits in swot_scores.items():
        pdf.add_section(f"{category} Scores", "\n".join([f"{trait}: {value:.2f}" for trait, value in traits.items()]))

    # Sentiment Analysis
    pdf.add_section("Sentiment Analysis", "\n".join([
        f"{category}: {sentiments[category]}" for category in sentiments.keys()
    ]))

    # Recommendations
    pdf.add_section("Recommendations", "\n".join(recommendations))

    # Leadership Growth Path
    pdf.add_section("Leadership Growth Path", "\n".join(growth_path))

    # SWOT Visualization Charts
    for chart_path in chart_paths:
        pdf.add_image(chart_path)

    # Save PDF
    pdf_output_path = "/tmp/professional_swot_report.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path

# Execution: Generate Professional PDF
if st.button("Generate Professional PDF Report"):
    st.markdown("### Generating your Professional PDF Report...")

    # Example data for demonstration purposes
    example_swot_scores = {
        "Strengths": {"Leadership": 0.9, "Vision": 0.8},
        "Weaknesses": {"Time Management": 0.6, "Micromanagement": 0.7},
        "Opportunities": {"Innovation": 0.8, "Inclusivity": 0.85},
        "Threats": {"Conflict Avoidance": 0.5, "Indecisiveness": 0.4},
    }
    example_sentiments = {
        "Strengths": [{"label": "POSITIVE", "score": 0.95}],
        "Weaknesses": [{"label": "NEGATIVE", "score": 0.65}],
        "Opportunities": [{"label": "POSITIVE", "score": 0.9}],
        "Threats": [{"label": "NEGATIVE", "score": 0.45}],
    }
    example_recommendations = [
        "Leverage your strength in Leadership to inspire your team.",
        "Work on improving Time Management to enhance productivity.",
        "Explore opportunities in Innovation for organizational growth.",
        "Mitigate risks in Conflict Avoidance by addressing disagreements constructively."
    ]
    example_growth_path = [
        "Step 1: Focus on enhancing Vision through strategic planning.",
        "Step 2: Develop stronger time management skills with dedicated tools.",
        "Step 3: Build inclusivity by fostering a culture of collaboration."
    ]
    example_chart_paths = [
        "/tmp/bar_chart.png",
        "/tmp/heatmap_chart.png"
    ]  # Replace with actual chart paths

    # Generate PDF
    pdf_path = generate_professional_pdf(
        example_swot_scores,
        example_sentiments,
        example_recommendations,
        example_growth_path,
        example_chart_paths
    )

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="Download Your Professional SWOT Report",
            data=pdf_file,
            file_name="SWOT_Leadership_Report.pdf",
            mime="application/pdf"
        )

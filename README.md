🚀 Multi-Text Classification NLP Project
🔗 Live Demo:

👉 https://final-nlp-project-edt9ewf8gqt6rxy7pddjo2.streamlit.app/

This project is a Multi-Class Text Classification System built with Machine Learning + NLP + Streamlit.
It classifies multiple sentences at once into categories such as:

🟢 Sports

🔵 Tech

🟣 Politics

🩺 Medical

🚗 Vehicles

🎓 Education

🎬 Entertainment

💰 Finance

The app is designed to be simple, interactive, user-friendly, and beginner-friendly.
------------------------------------------------------------------------------------
Multi-Text Classification NLP Project

This project is a simple NLP-based text classification system built using TF-IDF Vectorization, Linear Support Vector Machine (SVM), and Streamlit.
It allows users to enter multiple sentences, and the model will classify each sentence into categories such as Sports, Tech, Politics, Medical, Vehicles, Education, Entertainment, and Finance.

A live version of the project is available here:
https://final-nlp-project-edt9ewf8gqt6rxy7pddjo2.streamlit.app/

Overview

The goal of this project is to demonstrate how to build a basic text classification app using machine learning and deploy it on Streamlit Cloud.
Users can input several sentences (one per line), and the app will return the predicted category for each sentence.

Features

Multi-class text classification

Allows multiple sentences at once

Pre-trained model loaded from a pickle file

Clean and simple Streamlit user interface

Clear button to reset input

Examples of sentences provided in the app

Works directly in the browser through Streamlit Cloud

How It Works

Text Vectorization
The project uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert words into numerical vectors.

Model Training
The classifier used is LinearSVC (Support Vector Machine).
It is trained on a small set of predefined text samples grouped into categories.

Model Saving
The trained TF-IDF vectorizer and classifier are saved together in model.pkl using Python's pickle library.

Prediction in Streamlit
The user enters sentences, the app transforms them using the saved TF-IDF vectorizer, and the model predicts the category.

Each category has example sentences used during training.

Example Sentences Used

Sports: "Virat Kohli hits a century"

Tech: "Google launches a new AI tool"

Politics: "Prime Minister announces a new law"

Medical: "Doctor advises regular health checkup"

Vehicles: "Tesla releases its latest electric car"

Education: "New education policy introduced"

Entertainment: "The movie became a box office hit"

Finance: "Stock market reaches an all-time high"

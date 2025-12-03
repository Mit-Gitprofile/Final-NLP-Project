import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load the saved model
# -----------------------------
with open("modelss/model.pkl", "rb") as file:
    tfidf, model = pickle.load(file)

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="🤖 Multi-Text Classification App", layout="centered")

# -----------------------------
# App Title & Description
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>
        <img src='https://twemoji.maxcdn.com/v/latest/svg/1f916.svg' width='40'/>
        Multi-Text Classification App
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Classify sentences into categories like <b>Sports</b>, "
    "<b>Tech</b>, <b>Politics</b>, <b>Medical</b>, <b>Vehicles</b> and more.</p>",
    unsafe_allow_html=True
)

# -----------------------------
# Text Area
# -----------------------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

st.session_state.user_text = st.text_area(
    "Enter sentence for text classification:",
    value=st.session_state.user_text,
    height=120,
    placeholder="Type your sentence here..."
)

# -----------------------------
# Buttons
# -----------------------------
col1, col2 = st.columns([4, 1])
with col1:
    classify_button = st.button("🔮 Classify Sentence")
with col2:
    clear_button = st.button("🧹 Clear Box")

if clear_button:
    st.session_state.user_text = ""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ================================
# Function: extract important words
# ================================
def get_category_words(sent, class_index, top_n=10):
    """
    Returns top contributing words for a label that exist in the input sentence
    """
    estimator = model.calibrated_classifiers_[0].estimator
    coef = estimator.coef_[class_index]
    feature_names = tfidf.get_feature_names_out()
    
    # Split sentence into words
    sent_words = set([w.lower() for w in sent.split() if w.lower() not in ENGLISH_STOP_WORDS])
    
    # Get words with highest positive weight
    top_idx = np.argsort(coef)[-50:]  # get top 50 influential words
    detected_words = []
    for i in reversed(top_idx):
        word = feature_names[i]
        if word in sent_words:
            detected_words.append((word, float(coef[i])))
        if len(detected_words) >= top_n:
            break
    
    return detected_words

# -----------------------------
# Classification Logic
# -----------------------------
if classify_button:
    text = st.session_state.user_text.strip()

    if text == "":
        st.warning("Please enter some text!")
    else:
        st.markdown("### 📌 Input Sentence:")
        st.write(text)

        # Transform the sentence
        X_new = tfidf.transform([text])

        # Get probabilities for all classes
        probabilities = model.predict_proba(X_new)[0]
        classes = model.classes_

        # Filter relevant classes (probability > 10%)
        label_scores = [(lbl, score) for lbl, score in zip(classes, probabilities) if score > 0.10]
        label_scores = sorted(label_scores, key=lambda x: x[1], reverse=True)

        if not label_scores:
            st.info("No strong categories detected.")
        else:
            st.markdown("### 🎯 Detected Categories:")

            for label, score in label_scores:  # <-- use label_scores instead of filtered
                percent = round(score * 100, 2)
                class_index = list(classes).index(label)

                # Get only words from sentence that contribute to this label
                detected_words = get_category_words(text, class_index, top_n=5)  # <-- use text

                # Display the label and probability
                st.markdown(
                    f"""
                    <div style='background:#e8f5e9;padding:10px;border-radius:8px;
                    margin-bottom:10px;border-left:6px solid #43a047'>
                        <b>{label.capitalize()}</b> — {percent}%<br><br>
                        <b>📝 Words detected in sentence:</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if detected_words:
                    for word, val in detected_words:
                        st.markdown(
                            f"""
                            <div style='margin-left:20px;margin-bottom:4px;'>
                                🔹 <b>{word}</b> — weight {round(val,4)}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        "<div style='margin-left:20px;margin-bottom:4px;'>No key words from this category found in sentence.</div>",
                        unsafe_allow_html=True
                    )



# -----------------------------
# Footer
# -----------------------------
st.markdown("<p style='text-align:center;'>Made by Mit Patel 😎</p>", unsafe_allow_html=True)

    
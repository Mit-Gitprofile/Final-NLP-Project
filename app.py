import streamlit as st
import pickle

# -----------------------------
# Load the saved model
# -----------------------------
with open("F:\\Github_Desktop\\NLP-Netural-Language-Process\\Final Exam\\model.pkl", "rb") as file:
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
    "<p style='text-align:center;'>Classify multiple sentences into categories like <b>Sports</b>, "
    "<b>Tech</b>, <b>Politics</b>, <b>Medical</b>, <b>Vehicles</b> and more.</p>",
    unsafe_allow_html=True
)

# -----------------------------
# Example sentences
# -----------------------------
st.markdown("### ✨ Examples of sentences you can try:")
st.markdown(
    """
    ✔ **Sports** – Virat Kohli scores a century  
    ✔ **Tech** – Google launches a new AI feature  
    ✔ **Politics** – The government announces a new policy  
    ✔ **Medical** – Doctors recommend regular health checkups  
    ✔ **Vehicles** – Tesla unveils a new electric car  
    ✔ **Education** – University introduces a new online course  
    ✔ **Entertainment** – A new movie breaks box office records  
    ✔ **Finance** – Stock market rises after positive earnings  
    """
)

# -----------------------------
# Initialize session state
# -----------------------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# -----------------------------
# Text Area
# -----------------------------
st.session_state.user_text = st.text_area(
    "Enter sentence for text Classification:",
    value=st.session_state.user_text,
    height=100,
    placeholder="iphone 18 launching soon..."
)

# -----------------------------
# Buttons in columns
# -----------------------------
col1, col2 = st.columns([4, 1])

with col1:
    classify_button = st.button("🔮 Classify Group")

with col2:
    clear_button = st.button("🧹 Clear Box")

# -----------------------------
# Clear Button Logic
# -----------------------------
if clear_button:
    st.session_state.user_text = ""

# -----------------------------
# Classify Button Logic
# -----------------------------
if classify_button:
    if st.session_state.user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        sentences = st.session_state.user_text.split("\n")
        X_new = tfidf.transform(sentences)
        predictions = model.predict(X_new)

        st.markdown("### 📌 Classification Results:")

        for text, label in zip(sentences, predictions):
            st.markdown(
                f"""
                <div style='background-color:#d4edda; padding:10px; 
                border-radius:8px; box-shadow:2px 2px 5px #28a745; margin-bottom:5px'>
                    📝 <b>Text:</b> {text}<br>
                    🎯 <b>Category:</b> {label.capitalize()}
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown("<p style='text-align:center;'>Made by Mit Patel 😎</p>", unsafe_allow_html=True)

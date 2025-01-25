import streamlit as st
import pickle

# Load the pre-trained model and vectorizer
with open('spam.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

# Define the Streamlit app
st.title("SMS Spam Detection")
st.write("Enter an SMS message to check if it is spam or not.")

# Input box for the user to enter an SMS message
user_input = st.text_area("Type your message here:", height=150)

# Button to classify the message
if st.button("Classify"):
    if user_input.strip():
        # Transform and predict
        input_vectorized = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(input_vectorized)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        
        # Display result
        if prediction == 1:
            st.error(f"Result: {result}")
        else:
            st.success(f"Result: {result}")
    else:
        st.warning("Please enter a valid SMS message.")

# Optional styling
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

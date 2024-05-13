import streamlit as st
from inference import Inference

def main():
    st.title("Emotion Prediction App")
    
    inference = Inference()

    user_input = st.text_input("Enter a sentence:")

    if st.button("Predict Emotions"):
        predicted_emotions = inference.run(user_input)

        if predicted_emotions:
            st.success(f"Predicted emotions: {predicted_emotions}")
        else:
            st.warning("No emotions detected.")

if __name__ == "__main__":
    main()

import streamlit as st
import google.generativeai as genai


# keys
f = open(".keys/gemini-key.txt")
key = f.read()

# API key
genai.configure(api_key= key)

# Configure Model
model = genai.GenerativeModel(model_name="gemini-1.5-flash", 
                              system_instruction="""Analyze the submitted code and identify potential bugs,
                              errors, or areas of improvement.Generate Bug Report and Fixed Code only.""")


# streamlit
st.title("🤖 An AI Code Reviewer")

user_prompt = st.text_area("Enter Your Python Code here...")

if st.button("Generate"):
    st.subheader("Code Review")
    response = model.generate_content(user_prompt)
    st.write(response.text)
    



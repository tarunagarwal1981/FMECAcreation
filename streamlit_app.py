import streamlit as st
import pandas as pd
import openai
import os
import time
from openai.error import RateLimitError

# Function to get the OpenAI API key
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

# Function to process FMECA data using GPT-4
def process_fmeca_data(df):
    fmeca_text = df.to_string(index=False)
    
    # Simplified FMEA prompt to reduce token usage
    fmeca_training_prompt = """
    You are an experienced reliability engineer specialized in FMEA (Failure Mode and Effects Analysis). 
    I need you to process the following table of data and categorize it based on the following definitions:
    - **Failure Mode**: How the component fails. (e.g., "Piston ring fractures")
    - **Failure Symptom**: Observable indicators. (e.g., "Increased vibration")
    - **Failure Effect**: Impact on system performance. (e.g., "Reduced engine power")
    - **Failure Cause**: Why the failure occurs. (e.g., "Wear and tear")
    
    Format the data into four columns: Failure Mode, Symptom, Cause, and Effect.
    Here is the data:
    """

    retries = 5
    initial_wait = 5  # Start with 5 seconds wait time for exponential backoff
    for i in range(retries):
        try:
            # Send the data to GPT-4 for FMECA processing
            response = openai.Completion.create(
                engine="gpt-4",
                prompt=fmeca_training_prompt + fmeca_text,
                max_tokens=3000,
                temperature=0
            )
    
            # Extract the GPT-4 response
            output_text = response.choices[0].text.strip()

            # Assuming GPT-4 outputs a CSV-like structure, we convert it back into a DataFrame
            from io import StringIO

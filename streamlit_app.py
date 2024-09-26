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
    
    fmeca_training_prompt = """
    You are an experienced reliability engineer specialized in FMEA (Failure Mode and Effects Analysis). 
    I need you to process the following table of data and categorize it based on the following definitions:

    - **Failure Mode**: A specific combination of a component and a verb that describes how the component fails to perform its intended function. (Example: "Piston ring fractures")
    - **Failure Symptom**: An observable indicator that a failure mode is occurring or has occurred. (Example: "Increased vibration")
    - **Failure Effect**: The resulting impact or consequence of a failure mode on the system's performance. (Example: "Reduced engine power")
    - **Failure Cause**: The underlying reason or mechanism that leads to the occurrence of a failure mode. (Example: "Wear and tear")

    Training Guidelines:
    1. Identify failure modes as specific component-verb combinations.
    2. Recognize failure symptoms as observable indicators.
    3. Determine failure effects as impacts on system performance.
    4. Detect failure causes as underlying reasons or mechanisms.

    Please format the data into columns for:
    1. Failure Mode
    2. Failure Symptom
    3. Failure Cause
    4. Failure Effect

    Here is the data:
    """

    # Retry mechanism for RateLimitError
    retries = 5
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
            processed_df = pd.read_csv(StringIO(output_text))
            
            return processed_df
        
        except RateLimitError:
            if i < retries - 1:
                st.warning(f"Rate limit reached. Retrying in {2 ** i} seconds...")
                time.sleep(2 ** i)  # Exponential backoff: 2, 4, 8, 16 seconds
            else:
                st.error("Rate limit reached. Please try again later.")
                return None

# Set up OpenAI API key
openai.api_key = get_api_key()

# Streamlit UI layout
st.title('FMECA Data Processor with GPT-4')

# File uploader: Accepts Excel file from user
uploaded_file = st.file_uploader("Upload your FMECA Excel file", type=['xlsx'])

if uploaded_file is not None:
    # Read the Excel file
    try:
        df = pd.read_excel(uploaded_file)
        
        # Display the uploaded data
        st.write("Uploaded FMECA Data:")
        st.dataframe(df)

        # Process the data using GPT-4
        with st.spinner('Processing the data with GPT-4...'):
            processed_data = process_fmeca_data(df)

        # If processing was successful, display the processed data
        if processed_data is not None:
            st.write("Processed FMECA Data:")
            st.dataframe(processed_data)

            # Option to download the processed file
            st.download_button(
                label="Download Processed FMECA Data",
                data=processed_data.to_csv(index=False).encode('utf-8'),
                file_name="processed_fmeca.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")

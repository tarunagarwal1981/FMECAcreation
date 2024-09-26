import streamlit as st
import pandas as pd
import openai
import os
import time
from io import StringIO  # Move this import to the top of the file
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
            response = openai.ChatCompletion.create(
                engine="gpt-3.5-turbo",
                prompt=fmeca_training_prompt + fmeca_text,
                max_tokens=3000,
                temperature=0
            )
    
            # Extract the GPT-4 response
            output_text = response.choices[0].text.strip()

            # Convert the GPT-4 output into a DataFrame
            processed_df = pd.read_csv(StringIO(output_text))
            return processed_df
        
        except RateLimitError:
            if i < retries - 1:
                wait_time = initial_wait * (2 ** i)  # Exponential backoff: 5, 10, 20, 40 seconds
                st.warning(f"Rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error("Rate limit reached. Please try again later.")
                return None

# Throttle user requests to avoid rate limiting
def throttle_requests(threshold=60):
    if 'last_request_time' not in st.session_state:
        st.session_state.last_request_time = 0
    current_time = time.time()
    
    if current_time - st.session_state.last_request_time < threshold:
        st.warning(f"Please wait {threshold - (current_time - st.session_state.last_request_time):.0f} seconds before making another request.")
        return False
    
    st.session_state.last_request_time = current_time
    return True

# Caching the processed data to avoid redundant API calls
@st.cache_data(show_spinner=False)
def process_fmeca_data_cached(df):
    return process_fmeca_data(df)

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

        # Check if we should throttle requests
        if throttle_requests(threshold=60):
            # Process the data using GPT-4
            with st.spinner('Processing the data with GPT-4...'):
                processed_data = process_fmeca_data_cached(df)

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

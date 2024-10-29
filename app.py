import streamlit as st
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
import toml
import tempfile
from openai import OpenAI

def load_api_key(secret_file_path):
    """Load the OpenAI API key from a TOML file."""
    try:
        secrets = toml.load(secret_file_path)
        return secrets['OPENAI_API_KEY']
    except Exception as e:
        st.error(f"Error loading API key: {e}")
        return None

def initialize_openai_client(api_key):
    """Initialize the OpenAI client using the provided API key."""
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

def transcribe_audio(client, audio_file):
    """Transcribe the uploaded audio file using OpenAI's Whisper model."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name

        with open(tmp_file_path, "rb") as file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=file
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def extract_info(client, transcript, fields):
    """Extract information from the transcribed text using the GPT model."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(transcript)
        if fields:
            prompt = PromptTemplate(
                input_variables=["chunk", "fields"],
                template="Extract the following information from the text chunk: {fields}. Return the result as a JSON string. If the information is not found, return an empty JSON object. Text chunk: {chunk}"
            )
        else:
            prompt = PromptTemplate(
                input_variables=["chunk"],
                template="Extract important information from the following text chunk. Include any relevant details such as names, dates, locations, key points, or any other significant data. Return the result as a JSON string with appropriate keys for each piece of information. If no important information is found, return an empty JSON object. Text chunk: {chunk}"
            )

        results = []
        for chunk in chunks:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts information from text."},
                    {"role": "user", "content": prompt.format(chunk=chunk, fields=fields) if fields else prompt.format(chunk=chunk)}
                ],
                temperature=0
            )
            result = response.choices[0].message.content
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                st.warning(f"Failed to parse JSON from chunk. Raw response: {result}")
                parsed_result = {}
            results.append(parsed_result)

        # Merge results from all chunks
        merged_result = {}
        for result in results:
            merged_result.update(result)

        return merged_result
    except Exception as e:
        st.error(f"Error extracting information: {str(e)}")
        return None

def handle_file_upload(uploaded_file):
    """Handle the uploaded file and return its transcript."""
    try:
        if uploaded_file:
            st.audio(uploaded_file)
            return uploaded_file
        else:
            st.warning("Please upload a valid audio file.")
            return None
    except Exception as e:
        st.error(f"Error handling file upload: {e}")
        return None

def main():
    st.title("Audio Transcription and Information Extraction")
    # Add toml file containing openai api key 
    file_path = 'secrets.toml'
    # Load API key and initialize client
    api_key = load_api_key(file_path)
    if api_key is None:
        return
    client = initialize_openai_client(api_key)
    if client is None:
        return

    # File upload and user input handling
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    user_input = st.text_input("Enter fields to extract (comma-separated). Optional.",)

    # Transcribe and process the audio
    if uploaded_file is not None:
        audio_file = handle_file_upload(uploaded_file)
        if audio_file:
            transcript = transcribe_audio(client, audio_file)
            if transcript:
                st.subheader("Transcript")
                st.write(transcript)

                # Extract fields
                if user_input:
                    fields = [field.strip() for field in user_input.split(",")]
                else:
                    fields = []
                    
                if st.button("Extract Information"):
                    extracted_info = extract_info(client, transcript, fields)
                    if extracted_info:
                        st.subheader("Extracted Information")
                        st.json(extracted_info)

                        # Download option for JSON file
                        json_string = json.dumps(extracted_info, indent=2)
                        st.download_button(
                            label="Download JSON",
                            file_name="extracted_info.json",
                            mime="application/json",
                            data=json_string,
                        )

if __name__ == "__main__":
    main()

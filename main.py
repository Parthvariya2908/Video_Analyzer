# import streamlit as st
# from moviepy.video.io.VideoFileClip import VideoFileClip
# from nltk.tokenize import sent_tokenize
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone , ServerlessSpec
# import assemblyai as aai
# import requests
# import tempfile
# import nltk
# import os
# from nltk.data import find
# nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")

# try:
#     find('tokenizers/punkt')
# except LookupError:
#     # If 'punkt' is not found, download it
#     if not os.path.exists(nltk_data_dir):
#         os.makedirs(nltk_data_dir)
#     nltk.download('punkt', download_dir=nltk_data_dir)
# # API Keys
# ASSEMBLY_AI_API_KEY = "a8b29bde074f476688c91cfda9773078"
# HF_API_KEY = "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"
# PINECONE_API_KEY = "pcsk_6ga2M9_GTQwg6T5DdYkzQQDSrqhaLTP2Kk9MJzSEGavgVXquHZ3KN9rCAWP1n2JqeUX1E9"

# session_id = st.session_state.get('session_id', None)
# if session_id is None:
#     import uuid
#     session_id = str(uuid.uuid4())
#     st.session_state['session_id'] = session_id
# # print(session_id)
# PINECONE_INDEX_NAME = f"{session_id}-Interview"
# PINECONE_INDEX_NAME = PINECONE_INDEX_NAME.lower()  # Convert to lowercase
# PINECONE_INDEX_NAME = ''.join(c if c.isalnum() or c == '-' else '-' for c in PINECONE_INDEX_NAME)

# print(len(PINECONE_INDEX_NAME) )
# if len(PINECONE_INDEX_NAME) > 45:
#     PINECONE_INDEX_NAME = PINECONE_INDEX_NAME[:40]

# # HF_API_MODEL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
# HF_API_MODEL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
# # Initialize Pinecone
# pc = Pinecone(
#         api_key="pcsk_6ga2M9_GTQwg6T5DdYkzQQDSrqhaLTP2Kk9MJzSEGavgVXquHZ3KN9rCAWP1n2JqeUX1E9"
#     )
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#         pc.create_index(
#             name= PINECONE_INDEX_NAME,
#             dimension=384,  
#             metric='cosine',  
#             spec=ServerlessSpec(
#                 cloud='aws',
#                 region='us-east-1'  
#             )
#         )
# index = pc.Index(PINECONE_INDEX_NAME)

# # Function to convert video to audio
# # def video_to_audio(video_file, output_audio_path="output_audio.mp3"):
# #     video = VideoFileClip(video_file)
# #     audio = video.audio
# #     audio.write_audiofile(output_audio_path)
# #     audio.close()
# #     video.close()
# #     return output_audio_path

# def video_to_audio(uploaded_file):
#     # Create a temporary file to store the uploaded video
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(uploaded_file.read())  # Write the uploaded file's content to temp file
#         temp_video_path = temp_video.name  # Get the path of the temporary file
    
#     # Extract audio
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
#         video = VideoFileClip(temp_video_path)
#         audio = video.audio
#         audio.write_audiofile(temp_audio.name)
#         audio.close()
#         video.close()
#         audio_path = temp_audio.name  # Get the path of the temporary audio file
    
#     return audio_path

# # Function to transcribe audio to text
# def audio_to_text(audio_path):
#     aai.settings.api_key = ASSEMBLY_AI_API_KEY
#     transcriber = aai.Transcriber()
#     transcript = transcriber.transcribe(audio_path)
#     return transcript.text

# # Function to process text into chunks
# # def load_text_chunks(text):
# #     sentences = sent_tokenize(text)
# #     chunk_size = 15
# #     chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
# #     return chunks
# def load_text_chunks(text):
#     sentences = sent_tokenize(text)  # Tokenize the text into sentences
#     chunks = sentences  # Each sentence becomes a chunk
#     return chunks

# # Function to create embeddings and upsert into Pinecone
# def create_and_store_embeddings(chunks, session_id):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = []
#     for i, chunk in enumerate(chunks):
#         embedding_id = f"{session_id}-{i}"
#         embedding = model.encode(chunk).tolist()
#         index.upsert([(embedding_id, embedding, {"text": chunk})])
#         embeddings.append(embedding_id)
#     return embeddings

# # Function to retrieve relevant chunks from Pinecone
# def get_relevant_chunks(query, k=10):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     query_embedding = model.encode(query).tolist()
#     search_results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
#     return [result['metadata']['text'] for result in search_results['matches']]

# # Function to delete embeddings from Pinecone
# def remove_embeddings(session_ids):
#     for session_id in session_ids:
#         index.delete(filter={"id": {"$regex": f"^{session_id}-"}})

# # Hugging Face Query Handler
# def query_handler(query, context_text):
#     headers = {"Authorization": f"Bearer {HF_API_KEY}"}
#     prompt = f"give the  most relevent clean and accurate answer of the quation using the Context: {context_text} \n\n Question: {query}\nAnswer:"
#     response = requests.post(HF_API_MODEL, headers=headers, json={"inputs": prompt})
#     return response.json()[0]['generated_text']

# # Streamlit App
# st.title("Video Analysis & Q&A Tool with Embedding Cleanup")
# st.write("Upload a video to transcribe and perform question-answering based on its content.")

# # Generate a unique session ID
# session_id = st.session_state.get('session_id', None)
# if session_id is None:
#     import uuid
#     session_id = str(uuid.uuid4())
#     st.session_state['session_id'] = session_id

# uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "avi", "mkv"])
# if uploaded_file:
#     # Convert video to audio
#     st.info("Extracting audio from video...")
#     audio_path = video_to_audio(uploaded_file)

#     # Transcribe audio
#     st.info("Transcribing audio to text...")
#     transcript = audio_to_text(audio_path)
#     st.success("Transcription completed!")
#     st.write("Transcript:")
#     st.text_area("Transcript", transcript, height=300)

#     # Process text into chunks
#     st.info("Processing transcript and storing in Pinecone...")
#     chunks = load_text_chunks(transcript)

#     # Store chunks and embeddings in Pinecone
#     embedding_ids = create_and_store_embeddings(chunks, session_id)
#     st.session_state['embedding_ids'] = embedding_ids
#     st.success("Transcript stored in Pinecone!")

#     # Question Answering
#     query = st.text_input("Enter your question:")
#     if query:
#         st.info("Fetching the most relevant answer...")
#         relevant_chunks = get_relevant_chunks(query)
#         context_text = " ".join(relevant_chunks)
#         answer = query_handler(query, context_text)
#         st.success("Answer:")
#         st.write(answer)

# # Remove the index when session ends
# if st.button("End Session"):
#     if 'embedding_ids' in st.session_state:
#         remove_embeddings([session_id])
#         # Optionally delete the session-specific index
#         pc.delete_index(name = PINECONE_INDEX_NAME)  # Deletes the session-specific index
#         del st.session_state['embedding_ids']
#         st.success("Session ended and embeddings removed from Pinecone.")


import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone , ServerlessSpec
import assemblyai as aai
import requests
import tempfile
import nltk
import os
from nltk.data import find
from huggingface_hub import InferenceApi
from nltk.data import find
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")

try:
    nltk.download('punkt_tab')
except LookupError as e:
    print(f"Error: {e}")
    
try:
    find('tokenizers/punkt')
except LookupError:
    # If 'punkt' is not found, download it
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.download('punkt', download_dir=nltk_data_dir)

# API Keys
ASSEMBLY_AI_API_KEY = st.secrets["ASSEMBLY_AI_API_KEY"]
HF_API_KEY =st.secrets["HF_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone
pc = Pinecone(
        api_key=PINECONE_API_KEY
    )
# Session ID Management
session_id = st.session_state.get('session_id', None)
if session_id is None:
    import uuid
    session_id = str(uuid.uuid4())
    st.session_state['session_id'] = session_id

# Pinecone Index Name
PINECONE_INDEX_NAME = f"{session_id}-Interview".lower()
PINECONE_INDEX_NAME = ''.join(c if c.isalnum() or c == '-' else '-' for c in PINECONE_INDEX_NAME)
if len(PINECONE_INDEX_NAME) > 45:
    PINECONE_INDEX_NAME = PINECONE_INDEX_NAME[:40]

# Create Pinecone index if it does not exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # Embedding size
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  
        )
    )
index = pc.Index(PINECONE_INDEX_NAME)

# Hugging Face model for summarization
def summerizer(relevant_chunks):
    # Initialize summarization API
    retrieved_text = " ".join(relevant_chunks)
    summarizer = InferenceApi(repo_id="facebook/bart-large-cnn", token=HF_API_KEY)
    # Summarize the retrieved text
    response = summarizer(inputs=retrieved_text)
    summary = response[0]["summary_text"]
    return summary

# Function to convert video to audio
def video_to_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        video = VideoFileClip(temp_video_path)
        audio = video.audio
        audio.write_audiofile(temp_audio.name)
        audio.close()
        video.close()
        audio_path = temp_audio.name
    return audio_path

# Function to transcribe audio to text
def audio_to_text(audio_path):
    aai.settings.api_key = ASSEMBLY_AI_API_KEY
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path)
    return transcript.text

# Function to process text into chunks
def load_text_chunks(text):
    sentences = sent_tokenize(text)
    return sentences

# Function to create embeddings and upsert into Pinecone
def create_and_store_embeddings(chunks, session_id):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding_id = f"{session_id}-{i}"
        embedding = model.encode(chunk).tolist()
        index.upsert([(embedding_id, embedding, {"text": chunk})])
        embeddings.append(embedding_id)
    return embeddings

# Function to retrieve relevant chunks from Pinecone
def get_relevant_chunks(query, k=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).tolist()
    search_results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
    return [result['metadata']['text'] for result in search_results['matches']]

# Function to delete embeddings from Pinecone
def remove_embeddings(session_ids):
    for session_id in session_ids:
        index.delete(filter={"id": {"$regex": f"^{session_id}-"}})

# Hugging Face Query Handler for Q&A
def query_handler(query, context_text):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    prompt = f"give the most relevant clean and accurate answer of the question using the Context: {context_text} \n\n Question: {query}\nAnswer:"
    response = requests.post("https://api-inference.huggingface.co/models/google/flan-t5-large", headers=headers, json={"inputs": prompt})
    return response.json()[0]['generated_text']

# Streamlit App UI
st.title("Video Analysis & Q&A Tool with Embedding Cleanup")
st.write("Upload a video to transcribe and perform question-answering based on its content.")

uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "avi", "mkv"])

if uploaded_file:
    # Convert video to audio
    st.info("Extracting audio from video...")
    audio_path = video_to_audio(uploaded_file)

    # Transcribe audio to text
    st.info("Transcribing audio to text...")
    transcript = audio_to_text(audio_path)
    st.success("Transcription completed!")
    st.write("Transcript:")
    st.text_area("Transcript", transcript, height=300)

    # Process text into chunks
    st.info("Processing transcript and storing in Pinecone...")
    chunks = load_text_chunks(transcript)

    # Store chunks and embeddings in Pinecone
    embedding_ids = create_and_store_embeddings(chunks, session_id)
    st.session_state['embedding_ids'] = embedding_ids
    st.success("Transcript stored in Pinecone!")

    # Display the summary of the transcript
    summary = summerizer(chunks)
    st.write("Summary of the Transcript:")
    st.text_area("Summary", summary, height=200)

    # Question Answering
    query = st.text_input("Enter your question:")
    if query:
        st.info("Fetching the most relevant answer...")
        relevant_chunks = get_relevant_chunks(query)
        context_text = " ".join(relevant_chunks)
        answer = query_handler(query, context_text)
        st.success("Answer:")
        st.write(answer)

# Remove the index when session ends
if st.button("End Session"):
    try:
            pc.delete_index(name=PINECONE_INDEX_NAME)
            st.success(f"Session ended and index {PINECONE_INDEX_NAME} deleted.")
    except Exception as e:
            st.error(f"Error deleting index: {str(e)}")

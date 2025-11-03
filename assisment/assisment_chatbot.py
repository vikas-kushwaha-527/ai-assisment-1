import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import torch
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pinecone
from pinecone import Pinecone, ServerlessSpec
import hashlib
import time

load_dotenv()

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
huggingface_token = os.getenv("HF_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# --- Pinecone Helper Functions ---
@st.cache_resource
def init_pinecone():
    if pinecone_api_key:
        pc = Pinecone(api_key=pinecone_api_key)
        return pc
    return None

def get_or_create_index(pc, index_name="chatbot-rag", dimension=384):
    if pc:
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec = ServerlessSpec(
                    cloud = "aws",
                    region = "us-east-1"
                )
            )
        return pc.Index(index_name)
    return None

# --- RAG Helper Functions ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def find_relevant_chunks(query, chunks, embeddings, model, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def find_relevant_chunks_pinecone(query, model, index, top_k=3):
    if not index:
        return []
    
    query_embedding = model.encode([query]).tolist()[0]
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    return [match.metadata.get('text', '') for match in results.matches]

def store_chunks_in_pinecone(chunks, embeddings, index, file_name):
    if not index:
        return False
    
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = hashlib.md5(f"{file_name}_{i}_{chunk[:50]}".encode()).hexdigest()
        vectors.append({
            "id": chunk_id,
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk,
                "file_name": file_name,
                "chunk_index": i
            }
        })
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    
    return True

# --- LLM Interaction Function ---
def get_llm_response(selected_llm, messages, context=None):
    """
    Interacts with the selected LLM API and returns the response,
    using conversation history.

    Args:
        selected_llm: The name of the selected LLM (e.g., "Azure OpenAI", "Hugging Face Mistral").
        messages: A list of message dictionaries representing the conversation history.

    Returns:
        The response from the selected LLM.
    """
    if selected_llm == "Azure OpenAI":
        try:
            # Check if Azure OpenAI environment variables are set
            if not azure_openai_endpoint or not azure_openai_key or not azure_openai_deployment:
                 return "Azure OpenAI API key, endpoint, or deployment name not found in environment variables."

            # Initialize Azure OpenAI client
            client = AzureOpenAI(
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_key,
                api_version="2023-05-15" # Use a supported API version
            )

            # Format history for Azure OpenAI (list of message objects)
            # Ensure the first message is a system message if no history exists
            system_content = "You are a helpful assistant."
            if context:
                system_content += f" Use the following context to answer questions: {context}"
            formatted_messages = [{"role": "system", "content": system_content}] + \
                                 [{"role": m["role"], "content": m["content"]} for m in messages]


            # Create chat completion
            response = client.chat.completions.create(
                model=azure_openai_deployment,
                messages=formatted_messages
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Error interacting with Azure OpenAI: {e}"

    elif selected_llm == "Hugging Face Mistral":
        try:
            # Check if Hugging Face environment variable is set
            if not huggingface_token:
                return "Hugging Face API token not found in environment variables."

            # Hugging Face Inference API endpoint
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
            headers = {"Authorization": f"Bearer {huggingface_token}"}

            # Format history for Mistral
            formatted_input = "<s>"
            for i, message in enumerate(messages):
                if message["role"] == "user":
                    user_content = message['content']
                    if context and i == len(messages) - 1:  # Add context to the last user message
                        user_content = f"Context: {context}\n\nQuestion: {user_content}"
                    formatted_input += f"[INST] {user_content} [/INST]"
                elif message["role"] == "assistant":
                    formatted_input += f"{message['content']}</s>"

            # Prepare payload for the API
            payload = {
                "inputs": formatted_input,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "return_full_text": False  # Only return generated text, not the prompt
                }
            }

            # Make request to Hugging Face Inference API with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                
                # Check if request was successful
                if response.status_code == 200:
                    try:
                        # Check if response has content
                        if not response.text.strip():
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            return "Model is loading. Please try again in a few moments."
                        
                        result = response.json()
                        
                        # Handle model loading response
                        if isinstance(result, dict) and 'error' in result:
                            if 'loading' in result['error'].lower():
                                if attempt < max_retries - 1:
                                    time.sleep(10)  # Wait longer for model loading
                                    continue
                                return "Model is currently loading. Please try again in a few minutes."
                            return f"API Error: {result['error']}"
                        
                        # Extract generated text
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get('generated_text', '')
                        else:
                            generated_text = result.get('generated_text', '')
                        
                        # Clean up the response
                        generated_text = generated_text.strip()
                        
                        # Remove any trailing </s> if present
                        if generated_text.endswith("</s>"):
                            generated_text = generated_text[:-4].strip()
                        
                        return generated_text if generated_text else "No response generated. Please try again."
                        
                    except requests.exceptions.JSONDecodeError:
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return "Model is loading or temporarily unavailable. Please try again later."
                
                elif response.status_code == 503:
                    if attempt < max_retries - 1:
                        time.sleep(10)  # Wait longer for 503 errors
                        continue
                    return "Model is currently loading. Please try again in a few minutes."
                else:
                    try:
                        error_data = response.json()
                        error_message = error_data.get('error', 'Unknown error')
                    except:
                        error_message = f"HTTP {response.status_code}"
                    
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return f"Error from Hugging Face API: {error_message} (Status: {response.status_code})"
            
            return "Failed to get response after multiple attempts. Please try again later."

        except Exception as e:
            return f"Error interacting with Hugging Face Mistral: {e}"
    else:
        return "Invalid LLM selection."

# --- Streamlit Application Layout ---

st.title("Simple LLM Chatbot")

# Define available LLM options and get user selection
llm_options = ["Azure OpenAI", "Hugging Face Mistral"]
selected_llm = st.selectbox("Select your preferred LLM:", llm_options)

# RAG Configuration
use_rag = st.checkbox("Enable RAG (Retrieval-Augmented Generation)")
use_pinecone = st.checkbox("Use Pinecone for vector storage", help="Requires PINECONE_API_KEY in environment")

if use_rag:
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        if "document_processed" not in st.session_state or st.session_state.get("uploaded_file_name") != uploaded_file.name:
            with st.spinner("Processing document..."):
                # Extract text from PDF
                document_text = extract_text_from_pdf(uploaded_file)
                
                # Chunk the text
                chunks = chunk_text(document_text)
                
                # Load embedding model and create embeddings
                embedding_model = load_embedding_model()
                embeddings = embedding_model.encode(chunks)
                
                # Initialize Pinecone if enabled
                pinecone_index = None
                if use_pinecone:
                    pc = init_pinecone()
                    if pc:
                        pinecone_index = get_or_create_index(pc)
                        if pinecone_index:
                            store_success = store_chunks_in_pinecone(chunks, embeddings, pinecone_index, uploaded_file.name)
                            if store_success:
                                st.success("Document stored in Pinecone successfully!")
                            else:
                                st.warning("Failed to store in Pinecone, using local storage.")
                    else:
                        st.warning("Pinecone not configured. Check PINECONE_API_KEY environment variable.")
                
                # Store in session state
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.embedding_model = embedding_model
                st.session_state.pinecone_index = pinecone_index
                st.session_state.document_processed = True
                st.session_state.uploaded_file_name = uploaded_file.name
                
            st.success(f"Document '{uploaded_file.name}' processed successfully!")
        else:
            st.info(f"Document '{uploaded_file.name}' is already processed.")

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a text input field for the user to type their message
# st.chat_input handles the text input and the implicit send button (Enter key)
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response using the selected LLM and conversation history
    with st.chat_message("assistant"):
        # Create a placeholder for the assistant's response while generating
        with st.spinner("Generating response..."):
            context = None
            if use_rag and st.session_state.get("document_processed"):
                # Find relevant chunks for RAG
                if use_pinecone and st.session_state.get("pinecone_index"):
                    # Use Pinecone for retrieval
                    relevant_chunks = find_relevant_chunks_pinecone(
                        prompt,
                        st.session_state.embedding_model,
                        st.session_state.pinecone_index
                    )
                else:
                    # Use local embeddings
                    relevant_chunks = find_relevant_chunks(
                        prompt, 
                        st.session_state.chunks, 
                        st.session_state.embeddings, 
                        st.session_state.embedding_model
                    )
                context = "\n\n".join(relevant_chunks)
            
            # Pass the entire conversation history and context to the LLM function
            response = get_llm_response(selected_llm, st.session_state.messages, context)
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

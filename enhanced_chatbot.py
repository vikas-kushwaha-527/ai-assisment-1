import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import PyPDF2
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()

# Environment variables
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
huggingface_token = os.getenv("HF_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# --- Initialize Clients ---
@st.cache_resource
def init_hf_client():
    if huggingface_token:
        return InferenceClient(token=huggingface_token)
    return None

@st.cache_resource
def init_pinecone():
    if pinecone_api_key:
        return Pinecone(api_key=pinecone_api_key)
    return None

@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Pinecone Setup ---
def get_or_create_index(pc, index_name="chatbot-rag", dimension=384):
    if not pc:
        return None
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)  # Wait for index creation
    return pc.Index(index_name)

# --- Document Processing ---
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_documents_with_langchain(text, file_name):
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Create Document objects
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": file_name, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    return documents

# --- LLM Functions ---
def get_llm_response(selected_llm, messages, context=None):
    if selected_llm == "Azure OpenAI":
        try:
            if not all([azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
                return "Azure OpenAI credentials not found in environment variables."

            client = AzureOpenAI(
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_key,
                api_version="2023-05-15"
            )

            system_content = "You are a helpful assistant."
            if context:
                system_content += f" Use the following context to answer questions: {context}"
            
            formatted_messages = [{"role": "system", "content": system_content}] + messages

            response = client.chat.completions.create(
                model=azure_openai_deployment,
                messages=formatted_messages
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Error with Azure OpenAI: {e}"

    elif selected_llm == "Hugging Face Mistral":
        try:
            if not huggingface_token:
                return "Hugging Face token not found in environment variables."

            # Initialize client with specific model
            client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=huggingface_token)

            # Format messages for chat completion
            formatted_messages = [{"role": "system", "content": "You are a helpful assistant."}]
            
            # Add context to system message if available
            if context:
                formatted_messages[0]["content"] += f" Use the following context to answer questions: {context}"
            
            # Add conversation history
            formatted_messages.extend(messages)

            # Use chat_completion method
            response = client.chat_completion(
                messages=formatted_messages,
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message["content"]

        except Exception as e:
            return f"Error with Hugging Face: {e}"
    
    return "Invalid LLM selection."

# --- Streamlit App ---
st.title("LLM Chatbot with LangChain & Pinecone")

# LLM Selection
llm_options = ["Azure OpenAI", "Hugging Face Mistral"]
selected_llm = st.selectbox("Select LLM:", llm_options)

# RAG Configuration
use_rag = st.checkbox("Enable RAG")
use_pinecone = st.checkbox("Use Pinecone Vector Store", help="Requires PINECONE_API_KEY")

# Document Upload and Processing
if use_rag:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if "processed_file" not in st.session_state or st.session_state.processed_file != file_key:
            with st.spinner("Processing document..."):
                # Extract text
                document_text = extract_text_from_pdf(uploaded_file)
                
                # Process with LangChain
                documents = process_documents_with_langchain(document_text, uploaded_file.name)
                
                # Initialize embeddings
                embeddings = init_embeddings()
                
                # Setup vector store
                vectorstore = None
                if use_pinecone:
                    pc = init_pinecone()
                    if pc:
                        index = get_or_create_index(pc)
                        if index:
                            vectorstore = LangchainPinecone.from_documents(
                                documents=documents,
                                embedding=embeddings,
                                index_name="chatbot-rag"
                            )
                            st.success("Document stored in Pinecone!")
                        else:
                            st.error("Failed to create Pinecone index")
                    else:
                        st.error("Pinecone not configured")
                
                # Store in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.documents = documents
                st.session_state.embeddings = embeddings
                st.session_state.processed_file = file_key
                
            st.success(f"Processed '{uploaded_file.name}' successfully!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            context = None
            
            # Retrieve relevant context if RAG is enabled
            if use_rag and st.session_state.get("vectorstore"):
                try:
                    # Use LangChain vector store for similarity search
                    relevant_docs = st.session_state.vectorstore.similarity_search(
                        prompt, k=3
                    )
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                except Exception as e:
                    st.error(f"Error retrieving context: {e}")
            
            # Get LLM response
            response = get_llm_response(selected_llm, st.session_state.messages, context)
            st.markdown(response)

    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
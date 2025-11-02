import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import json
import time
from datetime import datetime

load_dotenv()

# Environment variables
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
huggingface_token = os.getenv("HF_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Career data
CAREER_DATA = {
    "roles": [
        {"title": "Data Scientist", "skills": ["Python", "Machine Learning", "Statistics", "SQL", "Pandas"], "level": "Mid-Senior", "growth": "High"},
        {"title": "Software Engineer", "skills": ["Programming", "Algorithms", "System Design", "Git", "Testing"], "level": "Entry-Senior", "growth": "High"},
        {"title": "AI Engineer", "skills": ["Deep Learning", "TensorFlow", "PyTorch", "MLOps", "Cloud"], "level": "Mid-Senior", "growth": "Very High"},
        {"title": "Product Manager", "skills": ["Strategy", "Analytics", "Communication", "Agile", "Market Research"], "level": "Mid-Senior", "growth": "Medium"},
        {"title": "DevOps Engineer", "skills": ["Docker", "Kubernetes", "CI/CD", "AWS", "Monitoring"], "level": "Mid-Senior", "growth": "High"},
        {"title": "Frontend Developer", "skills": ["React", "JavaScript", "CSS", "HTML", "UI/UX"], "level": "Entry-Senior", "growth": "Medium"},
        {"title": "Backend Developer", "skills": ["APIs", "Databases", "Server Architecture", "Security", "Microservices"], "level": "Entry-Senior", "growth": "High"},
        {"title": "Cybersecurity Analyst", "skills": ["Network Security", "Penetration Testing", "Risk Assessment", "Compliance"], "level": "Mid-Senior", "growth": "Very High"}
    ],
    "learning_paths": {
        "Data Scientist": ["Python Basics", "Statistics", "Pandas/NumPy", "Machine Learning", "Deep Learning", "MLOps"],
        "AI Engineer": ["Programming", "Math/Statistics", "Machine Learning", "Deep Learning Frameworks", "MLOps", "Cloud Deployment"],
        "Software Engineer": ["Programming Fundamentals", "Data Structures", "Algorithms", "System Design", "Testing", "Version Control"]
    }
}

# Initialize clients
@st.cache_resource
def init_azure_client():
    if all([azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
        return AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version="2023-05-15"
        )
    return None

@st.cache_resource
def init_hf_client():
    if huggingface_token:
        return InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=huggingface_token)
    return None

@st.cache_resource
def init_pinecone():
    if pinecone_api_key:
        return Pinecone(api_key=pinecone_api_key)
    return None

@st.cache_resource
def init_embeddings():
    return SentenceTransformer('all-MiniLM-L6-v2')

def setup_vector_store():
    pc = init_pinecone()
    if not pc:
        return None
    
    index_name = "career-guidance"
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
    
    # Get index and embedding model
    index = pc.Index(index_name)
    embedding_model = init_embeddings()
    
    # Create and store embeddings for career data
    vectors = []
    for i, role in enumerate(CAREER_DATA["roles"]):
        content = f"Role: {role['title']}\nSkills: {', '.join(role['skills'])}\nLevel: {role['level']}\nGrowth: {role['growth']}"
        embedding = embedding_model.encode([content])[0]
        
        vectors.append({
            "id": f"role_{i}",
            "values": embedding.tolist(),
            "metadata": {"content": content, **role}
        })
    
    # Upsert vectors to Pinecone
    index.upsert(vectors=vectors)
    
    return {"index": index, "model": embedding_model}

def get_llm_response(messages, context=None, llm_choice="Azure OpenAI"):
    if llm_choice == "Azure OpenAI":
        client = init_azure_client()
        if not client:
            return "Azure OpenAI not configured"
        
        system_content = "You are a career guidance counselor. Provide personalized advice based on user goals and market trends."
        if context:
            system_content += f" Use this context: {context}"
        
        formatted_messages = [{"role": "system", "content": system_content}] + messages
        
        try:
            response = client.chat.completions.create(
                model=azure_openai_deployment,
                messages=formatted_messages,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
    
    else:  # Hugging Face
        client = init_hf_client()
        if not client:
            return "Hugging Face not configured"
        
        formatted_messages = [{"role": "system", "content": "You are a career guidance counselor."}]
        if context:
            formatted_messages[0]["content"] += f" Context: {context}"
        formatted_messages.extend(messages)
        
        try:
            response = client.chat_completion(
                messages=formatted_messages,
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"Error: {e}"

def find_matching_roles(user_input, vectorstore, top_k=3):
    if not vectorstore:
        return []
    
    try:
        # Get query embedding
        query_embedding = vectorstore["model"].encode([user_input])[0]
        
        # Search in Pinecone
        results = vectorstore["index"].query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract role metadata
        roles = []
        for match in results.matches:
            metadata = match.metadata.copy()
            metadata.pop('content', None)  # Remove content field
            roles.append(metadata)
        
        return roles
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def generate_learning_roadmap(role_title):
    if role_title in CAREER_DATA["learning_paths"]:
        return CAREER_DATA["learning_paths"][role_title]
    return ["Research role requirements", "Identify key skills", "Create learning plan", "Practice projects", "Build portfolio"]

def generate_newsletter():
    trends = [
        "AI/ML continues to dominate job market with 40% growth",
        "Remote work skills increasingly valued by employers",
        "Cloud computing expertise in high demand across industries",
        "Cybersecurity roles showing 25% salary increase",
        "Data science roles evolving to include MLOps skills"
    ]
    
    newsletter = f"""
    📰 **Career Insights Newsletter - {datetime.now().strftime('%B %Y')}**
    
    🔥 **Top Trends:**
    """
    
    for i, trend in enumerate(trends, 1):
        newsletter += f"\n{i}. {trend}"
    
    newsletter += """
    
    💡 **Recommendation:** Focus on AI/ML skills and cloud platforms for maximum career growth.
    
    📈 **Hot Skills This Month:** Python, AWS, Machine Learning, React, Kubernetes
    """
    
    return newsletter

# Streamlit App
st.set_page_config(page_title="AI Career Guidance", page_icon="🚀", layout="wide")

st.title("🚀 AI Career Guidance Chatbot")
st.markdown("*Your personalized 24×7 career mentor powered by AI*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    llm_choice = st.selectbox("Select LLM:", ["Azure OpenAI", "Hugging Face Mistral"])
    
    st.header("🎯 Quick Actions")
    if st.button("📊 Generate Newsletter", key="btn_newsletter"):
        st.session_state.show_newsletter = True
    
    if st.button("🔍 Explore Roles", key="btn_explore"):
        st.session_state.show_roles = True
    
    st.header("📈 Career Stats")
    st.metric("Available Roles", len(CAREER_DATA["roles"]))
    st.metric("Learning Paths", len(CAREER_DATA["learning_paths"]))

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hi! I'm your AI Career Guidance Assistant. Tell me about your career goals, current skills, or what you'd like to explore!"
    })

if "vectorstore" not in st.session_state:
    with st.spinner("Setting up career database..."):
        st.session_state.vectorstore = setup_vector_store()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Career Conversation")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about careers, skills, or roadmaps..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Find matching roles
                matching_roles = find_matching_roles(prompt, st.session_state.vectorstore)
                context = ""
                
                if matching_roles:
                    context = f"Relevant roles: {json.dumps(matching_roles[:2])}"
                
                # Get LLM response
                response = get_llm_response(st.session_state.messages, context, llm_choice)
                st.markdown(response)
                
                # Show role recommendations if relevant
                if matching_roles and any(word in prompt.lower() for word in ["role", "job", "career", "position"]):
                    st.markdown("### 🎯 Recommended Roles:")
                    for i, role in enumerate(matching_roles[:2]):
                        with st.expander(f"📋 {role['title']}"):
                            st.write(f"**Skills:** {', '.join(role['skills'])}")
                            st.write(f"**Level:** {role['level']}")
                            st.write(f"**Growth:** {role['growth']}")
                            
                            if st.button(f"Get Roadmap", key=f"roadmap_{i}_{role['title'].replace(' ', '_')}"):
                                roadmap = generate_learning_roadmap(role['title'])
                                st.write("**Learning Roadmap:**")
                                for j, step in enumerate(roadmap, 1):
                                    st.write(f"{j}. {step}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.header("📊 Insights")
    
    # Newsletter section
    if st.session_state.get("show_newsletter"):
        st.subheader("📰 Career Newsletter")
        newsletter = generate_newsletter()
        st.markdown(newsletter)
        if st.button("Close Newsletter", key="close_newsletter"):
            st.session_state.show_newsletter = False
    
    # Roles explorer
    elif st.session_state.get("show_roles"):
        st.subheader("🔍 Available Roles")
        for role in CAREER_DATA["roles"]:
            with st.expander(role["title"]):
                st.write(f"**Growth:** {role['growth']}")
                st.write(f"**Level:** {role['level']}")
                st.write(f"**Key Skills:** {', '.join(role['skills'][:3])}")
        if st.button("Close Roles", key="close_roles"):
            st.session_state.show_roles = False
    
    else:
        st.subheader("🎯 Career Focus Areas")
        focus_areas = ["AI/ML", "Software Development", "Data Science", "Cybersecurity", "Cloud Computing"]
        for i, area in enumerate(focus_areas):
            if st.button(area, key=f"focus_area_{i}"):
                st.session_state.messages.append({"role": "user", "content": f"Tell me about {area} careers"})
                st.rerun()
        
        st.subheader("📈 Market Trends")
        st.info("🔥 AI/ML roles growing 40% YoY")
        st.info("☁️ Cloud skills in high demand")
        st.info("🔒 Cybersecurity salaries up 25%")

# Footer
st.markdown("---")
st.markdown("*Powered by Azure OpenAI & Pinecone | Built for personalized career guidance*")
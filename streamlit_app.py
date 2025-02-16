import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from docx import Document
from pinecone import Pinecone, ServerlessSpec
import io

# Load environment variables
load_dotenv()

# Access API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "active-cyber"

# Initialize Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
index = pc.Index(INDEX_NAME)

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def generate_embedding(text):
    return embeddings.embed_query(text)

def split_document(document_text):
    paragraphs = document_text.split('\n')
    half = len(paragraphs) // 2
    return '\n'.join(paragraphs[:half]), '\n'.join(paragraphs[half:])

def upsert_document(document_text, metadata):
    document_size = len(document_text.encode('utf-8'))
    if document_size <= 40 * 1024:  # 40 KB
        embedding = generate_embedding(document_text)
        if embedding:
            document_id = metadata['title']
            metadata['text'] = document_text
            index.upsert(vectors=[(document_id, embedding, metadata)])
            return [document_id]
    else:
        chunk1, chunk2 = split_document(document_text)
        embedding1 = generate_embedding(chunk1)
        embedding2 = generate_embedding(chunk2)
        if embedding1 and embedding2:
            document_id1 = f"{metadata['title']}_chunk1"
            document_id2 = f"{metadata['title']}_chunk2"
            metadata1 = metadata.copy()
            metadata2 = metadata.copy()
            metadata1['text'] = chunk1
            metadata2['text'] = chunk2
            index.upsert(vectors=[
                (document_id1, embedding1, metadata1),
                (document_id2, embedding2, metadata2)
            ])
            return [document_id1, document_id2]
    return []

def query_pinecone(query):
    query_embedding = generate_embedding(query)
    if query_embedding:
        result = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        return [(match['id'], match['metadata']['text']) for match in result['matches']]
    else:
        return []

def get_answer(context, user_query):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    
    system_message = SystemMessage(content="""You are an AI assistant that provides information based on documents stored in a Pinecone vector database.
    When a user asks a question, the system retrieves the most relevant content from the top matching documents in Pinecone.
    Your task is to:
    1. Answer the user's question using ONLY the provided context.
    2. Provide ONLY the necessary information that directly answers the user's question.
    3. Be concise and avoid including any irrelevant or unnecessary details.
    4. Do not make up or infer any information beyond what is explicitly stated in the context.
    5.Remember, brevity and relevance are key. Stick strictly to addressing the user's specific query.""")
    human_message = HumanMessage(content=f"Context: {context}\n\nQuestion: {user_query}\n\nPlease provide a comprehensive answer based on the given context.")
    
    response = chat([system_message, human_message])
    return response.content

def main():
    st.set_page_config(page_title="Document Assistant", layout="wide")

    st.title("Document Assistant")

    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Upload DOCX Files", type="docx", accept_multiple_files=True)
        
        if uploaded_files and st.button("Upsert Documents"):
            with st.spinner("Uploading documents into pinecone index"):
                for uploaded_file in uploaded_files:
                    document_text = extract_text_from_docx(uploaded_file)
                    metadata = {"title": uploaded_file.name}
                    document_ids = upsert_document(document_text, metadata)
                    if document_ids:
                        st.success(f"Uploaded: {uploaded_file.name} (IDs: {', '.join(document_ids)})")
                    else:
                        st.error(f"Failed to upload: {uploaded_file.name}")

    # Main area for query interface
    st.header("Query Documents")
    user_query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Searching for the best answer..."):
                matches = query_pinecone(user_query)
                if matches:
                    context = " ".join([text for _, text in matches])
                    answer = get_answer(context, user_query)
                    st.write(answer)
                else:
                    st.warning("No relevant documents found. Please try a different question or upload more documents.")
        else:
            st.warning("Please enter a question before searching.")

if __name__ == "__main__":
    main()

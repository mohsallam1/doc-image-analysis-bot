import streamlit as st
import ollama
import os
import tempfile
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class DocumentProcessor:
    """Handles document processing for PDF and CSV files."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    def process_pdf(self, file_path: str) -> FAISS:
        """Process PDF and create vector store."""
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Extract and clean text
            texts = []
            for doc in documents:
                text = doc.page_content
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
            
            if not texts:
                raise ValueError("No text content found in PDF")
            
            # Split texts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            all_texts = text_splitter.split_text("\n".join(texts))
            
            # Create embeddings and vectorstore
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_texts(
                texts=all_texts,
                embedding=embedding_model
            )
            
            return vectorstore
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            raise
    
    def process_csv(self, file_path: str) -> FAISS:
        """Process CSV and create vector store."""
        try:
            # Read CSV using pandas
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to text format
            texts = []
            for _, row in df.iterrows():
                text = " | ".join(f"{col}: {val}" for col, val in row.items())
                texts.append(text)
            
            if not texts:
                raise ValueError("No content found in CSV")
            
            # Create vector store
            splits = self.text_splitter.split_text("\n".join(texts))
            vectorstore = FAISS.from_texts(splits, self.embeddings)
            
            return vectorstore
            
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            raise

class ImageCaptioner:
    """Handles image captioning using BLIP model."""
    
    def __init__(self):
        try:
            # Initialize BLIP model
            model_name = "Salesforce/blip-image-captioning-base"
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            
            # Enable GPU if available
            if torch.cuda.is_available():
                self.model.to('cuda')
                st.success("✅ GPU acceleration enabled for image processing!")
            
        except Exception as e:
            st.error(f"Error initializing image captioner: {str(e)}")
            raise
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for an image."""
        try:
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Use GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate caption
            output = self.model.generate(**inputs, max_new_tokens=100)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            st.error(f"Error generating caption: {str(e)}")
            return "Error processing image"

class Chatbot:
    """Main chatbot class for document Q&A."""
    
    def __init__(self):
        self.message_history: List[Dict] = []
        self.document_processor = DocumentProcessor()
        self.image_captioner = ImageCaptioner()
        self.model_name = "llama2"
        self.ensure_ollama_available()
    
    def ensure_ollama_available(self):
        """Ensure Ollama model is available."""
        try:
            models = ollama.list()
            model_exists = any(model.get('name') == self.model_name 
                             for model in models.get('models', []))
            
            if not model_exists:
                st.warning(f"Downloading {self.model_name} model... Please wait...")
                ollama.pull(self.model_name)
                st.success(f"✅ {self.model_name} model ready!")
                
        except Exception as e:
            st.error("⚠️ Error: Ollama not available. Please install Ollama first.")
            st.info("Installation guide: https://ollama.ai/")
            raise
    
    def setup_chain(self, vectorstore: FAISS):
        """Setup retrieval chain with vector store."""
        try:
            from langchain.chains import RetrievalQA
            from langchain_community.llms import Ollama
            
            # Initialize Ollama
            llm = Ollama(model=self.model_name)
            
            # Create retrieval chain
            self.chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                verbose=True
            )
            
        except Exception as e:
            st.error(f"Error setting up chain: {str(e)}")
            raise
    
    def process_query(self, query: str) -> str:
        """Process a text query."""
        try:
            if hasattr(self, 'chain'):
                # Use retrieval chain
                response = self.chain({"query": query})
                answer = response.get('result', 'No response generated')
            else:
                # Use direct LLM
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": query}]
                )
                answer = response['message']['content']
            
            # Update history
            self.message_history.extend([
                HumanMessage(content=query),
                AIMessage(content=answer)
            ])
            
            return answer
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

def initialize_session_state():
    """Initialize session state variables."""
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = Chatbot()
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.stop()
    
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Document & Image Analysis Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("📚 Document & Image Analysis Bot")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for uploads
    with st.sidebar:
        st.header("📎 Upload Files")
        
        # Document upload
        uploaded_doc = st.file_uploader(
            "Upload Document (PDF/CSV)",
            type=['pdf', 'csv'],
            key="doc_uploader"
        )
        
        if uploaded_doc and not st.session_state.document_processed:
            with st.spinner("Processing document..."):
                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_doc.type.split('/')[-1]}") as temp_file:
                        temp_file.write(uploaded_doc.getvalue())
                        temp_path = temp_file.name
                    
                    # Process document based on type
                    if uploaded_doc.type == "application/pdf":
                        vectorstore = st.session_state.chatbot.document_processor.process_pdf(temp_path)
                    else:
                        vectorstore = st.session_state.chatbot.document_processor.process_csv(temp_path)
                    
                    # Setup chain
                    st.session_state.chatbot.setup_chain(vectorstore)
                    st.session_state.document_processed = True
                    st.success("✅ Document processed successfully!")
                    
                    # Cleanup
                    os.unlink(temp_path)
                    
                except Exception as e:
                    st.error(f"Failed to process document: {str(e)}")
        
        # Image upload
        st.header("🖼️ Upload Image")
        uploaded_image = st.file_uploader(
            "Upload Image",
            type=['png', 'jpg', 'jpeg'],
            key="image_uploader"
        )
        
        if uploaded_image:
            try:
                # Process image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with st.spinner("Generating caption..."):
                    caption = st.session_state.chatbot.image_captioner.generate_caption(image)
                    st.success("✅ Caption generated!")
                    st.write(f"🎯 Caption: {caption}")
                    
            except Exception as e:
                st.error(f"Failed to process image: {str(e)}")
    
    # Chat interface
    st.header("💬 Chat")
    
    # Display message history
    for message in st.session_state.chatbot.message_history:
        with st.chat_message(message.type):
            st.write(message.content)
    
    # Chat input
    if prompt := st.chat_input("Ask about your document or type a message..."):
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.process_query(prompt)
            st.write(response)

if __name__ == "__main__":
    main()
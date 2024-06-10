from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
import warnings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_google_api_key():
    return os.getenv("AIzaSyBg9Hq7avlD4iX94pnU9ce6YwT1X5LPeVc")

# Import gevent and monkey-patch early to avoid MonkeyPatchWarning
import gevent.monkey
gevent.monkey.patch_all()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the Flask app
app = Flask(__name__)

def user_input(user_question):
    # Initialize Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_google_api_key())
    # Load FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    logger.info("-------------------------DATABASE LOADED!!!--------------------------")    
    # Search for similar documents
    docs = new_db.similarity_search(user_question)
    logger.info("-------------------------RETRIEVED SIMILAR DATA!!!--------------------------")        
    context = " ".join([doc.page_content for doc in docs])
    return context


# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Define the ask route to handle POST requests
@app.route('/ask', methods=['POST'])
def ask():
    # Get user's question from the request
    user_question = request.form['question']
    logger.info(f"USER QUESTION: {user_question}")
    # Get response based on user's question
    response = user_input(user_question)
    logger.info(f"User Question: {user_question}, Response: {response}")
    # Return the response as JSON
    return jsonify({'response': prettify_text(response)})

# Utility function to prettify text
def prettify_text(text):
    prettified = text.replace('\n', '<br>')
    prettified = prettified.replace('**', '<b>').replace('*', '<li>')
    prettified = prettified.replace('<b>', '</b>', 1)  # Ensure to close the first bold tag correctly
    return prettified

# Run the Flask app
if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True, threaded=True, port=5000, host='0.0.0.0')

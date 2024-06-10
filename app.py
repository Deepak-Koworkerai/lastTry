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
import requests
import ssl 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Set the SSL context to avoid verification issues within the Flask app context
ssl._create_default_https_context = ssl._create_unverified_context

def get_google_api_key():
    return os.getenv("AIzaSyBg9Hq7avlD4iX94pnU9ce6YwT1X5LPeVc")

# Import gevent and monkey-patch early to avoid MonkeyPatchWarning
import gevent.monkey
gevent.monkey.patch_all()


# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def user_input(user_question):
    # Initialize Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_google_api_key())
    # Load FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    logger.info("-------------------------DATABASE LOADED!!!--------------------------")    
    # Search for similar documents
    docs = new_db.similarity_search(user_question,k=1)
    logger.info("-------------------------RETRIEVED SIMILAR DATA!!!--------------------------")        
    context = " ".join([doc.page_content for doc in docs])
    return context


# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Define the ask route to handle POST requests
# @app.route('/ask', methods=['POST'])
# def ask():
#     # Get user's question from the request
#     user_question = request.form['question']
#     logger.info(f"USER QUESTION: {user_question}")
#     # Get response based on user's question
#     response = user_input(user_question)
#     logger.info(f"User Question: {user_question}, Response: {response}")
#     # Return the response as JSON
#     return jsonify({'response': prettify_text(response)})

@app.route('/ask', methods=['POST'])
def ask():
    # Get user's question from the request
    user_question = request.form['question']
    response = user_input(user_question)
    logger.info(f"User Question: {user_question}, Response: {response}")
    
    with app.app_context():
        # Prepare data to send to the other Flask application
        data = {
            'user_question': user_question,
            'response': response
        }
    
        other_app_url = 'https://embeddings-yijx.onrender.com/model'  # Replace with the actual URL of the other Flask application
        response = requests.post(other_app_url, json=data)
        logger.info("\n\n\n\n --------------RESPONSE--------------------------------\n\n\n")
        logger.info(type(response),"\n\n")
        out = response.json()['output']
        if response.status_code == 200:
            return jsonify({'response': prettify_text(out)})
        else:
            return jsonify({'success': False, 'error': 'Failed to get response from the other Flask application'})

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

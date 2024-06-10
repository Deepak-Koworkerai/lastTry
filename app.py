from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
import warnings

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

def get_conversational_chain():
    # Define the prompt template for the conversational chain
    prompt_template = """
    You are an AI friend designed to help Deepak by providing detailed and accurate answers based on the provided context. You are more than an assistant; you are a friend to Deepak. Ensure your responses are informative, contextually relevant, and align with Deepak's tone and style of communication.
    Converse like a human 
    Context:\n{context}\n

    Question:\n{question}\n

    Note:
    - If the question is directly related to the provided data, provide a detailed and accurate answer.
    - If the question pertains to a general topic or is conversational in nature, respond in a friendly, human-like manner. For example, for questions like "tell me something you know" or "is this the right time to talk," answer conversationally and encourage engagement.
    - If the answer is not present in the provided context and the question seems personal, unknown, or too common, respond by acknowledging your limitation in a friendly way. For instance, say "Oh no! I am not really aware of it, I shall ask Deepak and let you know later!!" to avoid giving incorrect information.
    - Always prioritize clarity, accuracy, and a friendly tone in your responses. Even if the answer is not relevant, try to respond in a helpful and engaging manner.

    Answer:
    """
    # Initialize the Google Generative AI model for conversation
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyBg9Hq7avlD4iX94pnU9ce6YwT1X5LPeVc')
    # Define the prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load the conversational chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # Initialize Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyBg9Hq7avlD4iX94pnU9ce6YwT1X5LPeVc')
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
    app.run(debug=True, threaded=True, port=5000, host='0.0.0.0', request_handler_class="gevent.pywsgi.WSGIHandler", timeout=150)

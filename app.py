from flask import Flask, request, jsonify, render_template
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
import time
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from google.api_core.exceptions import DeadlineExceeded
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



def get_conversational_chain():
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

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyBg9Hq7avlD4iX94pnU9ce6YwT1X5LPeVc')
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyBg9Hq7avlD4iX94pnU9ce6YwT1X5LPeVc')
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    response = user_input(user_question)
    return jsonify({'response': prettify_text(response)})

def prettify_text(text):
    prettified = text.replace('\n', '<br>')
    prettified = prettified.replace('**', '<b>').replace('*', '<li>')
    prettified = prettified.replace('<b>', '</b>', 1)  # Ensure to close the first bold tag correctly
    return prettified

if __name__ == '__main__':
    app.run(debug=True)
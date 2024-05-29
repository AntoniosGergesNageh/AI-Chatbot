from flask import Flask, request
from PyPDF2 import PdfReader
import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time

app = Flask(__name__)

def convert_pdf_to_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def load_data(text):
    return [bs4.BeautifulSoup(text, 'html.parser')]

def create_vectorstore(text_data):
    embeddings = OllamaEmbeddings(model="llama3")
    docs = load_data(text_data)
    return Chroma.from_documents(documents=docs, embedding=embeddings)

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

def combine_docs(docs, input_text):
    contents = []
    for doc in docs:
        content = doc.page_content
        if content is not None:
            contents.append(content)
    input_text_weight = 1
    contents.insert(0, input_text * input_text_weight)
    return "\n\n".join(contents)

def rag_chain(vectorstore, text, question):
    docs = load_data(text)
    formatted_context = combine_docs(docs, text)
    return ollama_llm(question, formatted_context)

@app.route('/')
def index():
    # Your HTML code here
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>RAG Chatbot</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #222;
                color: #ffc107;
            }
            .container {
                padding: 40px;
                max-width: 1200px;
                margin: 0 auto;
            }
            .title-container {
                text-align: center;
                margin-bottom: 30px;
            }
            .title-text {
                font-size: 40px;
                font-weight: bold;
                text-transform: uppercase;
                color: #ffc107;
                background: linear-gradient(45deg, #3498db, #ff7f00);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: inline-block;
            }
            .title-icon {
                font-size: 30px;
                color: #ffc107;
                margin-right: 10px;
            }
            .input-box label {
                color: #ffc107;
                font-size: 20px;
                font-weight: bold;
                text-transform: uppercase;
                margin-bottom: 5px;
                display: inline-block;
                padding: 5px 10px;
                background-color: #444;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
            }
            .input-box input[type="text"],
            .input-box textarea,
            .output-box {
                width: 100%;
                padding: 10px;
                border: 2px solid #444;
                border-radius: 5px;
                background-color: #444;
                color: #ccc;
                font-weight: bold;
            }
            .btn-primary {
                background-color: #ffc107;
                border: none;
                color: #222;
                font-weight: bold;
                text-transform: uppercase;
                padding: 10px 20px;
                margin-top: 10px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
            }
            .btn-primary:hover {
                background-color: #ffaa00;
            }
            .output-label {
                font-size: 20px;
                font-weight: bold;
                margin-top: 20px;
                color: #ffc107;
                text-transform: uppercase;
            }
            .output-box {
                margin-top: 10px;
                padding: 20px;
                background-color: #444;
                border-radius: 5px;
                color: #ccc; /* Change font color */
            }
            .timer {
                text-align: center;
                color: #ffc107;
                margin-top: 10px;
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="title-container">
                <span class="title-icon"><i class="fas fa-robot"></i></span>
                <h1 class="title-text">RAG Chatbot</h1>
            </div>
            <div class="row">
                <div class="col-lg-8">
                    <div class="input-box">
                        <label for="text-input">Enter Text:</label>
                        <textarea id="text-input" class="form-control" rows="15"></textarea>
                    </div>
                    <div class="input-box">
                        <label for="pdf-path">Enter PDF Path:</label>
                        <input type="text" id="pdf-path" class="form-control">
                    </div>
                    <div class="input-box">
                        <label for="question-input">Enter Question:</label>
                        <input type="text" id="question-input" class="form-control">
                    </div>
                    <button class="btn btn-primary btn-block mt-3" onclick="submitQuery()">Send</button>
                </div>
                <div class="col-lg-4">
                    <div class="output-label">Output:</div>
                    <div id="output-box" class="output-box">
                        <!-- Output will be displayed here -->
                    </div>
                    <div id="timer" class="timer" style="display:none;"></div>
                </div>
            </div>
        </div>
        <script>
            var timerInterval;
            function submitQuery() {
                clearInterval(timerInterval); // Clear previous interval
                var text = document.getElementById("text-input").value;
                var pdfPath = document.getElementById("pdf-path").value;
                var question = document.getElementById("question-input").value;
                document.getElementById("output-box").style.display = "none";
                document.getElementById("timer").style.display = "block";
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "/submit", true);
                xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
                xhr.onreadystatechange = function () {
                    if (xhr.readyState == 4 && xhr.status == 200) {
                        document.getElementById("output-box").innerHTML = formatOutput(xhr.responseText);
                        document.getElementById("output-box").style.display = "block";
                        clearInterval(timerInterval); // Stop the timer
                    }
                };
                var data = "text=" + encodeURIComponent(text) + "&pdf_path=" + encodeURIComponent(pdfPath) + "&question=" + encodeURIComponent(question);
                xhr.send(data);
                var startTime = new Date().getTime();
                timerInterval = setInterval(function() {
                    var elapsedTime = new Date().getTime() - startTime;
                    document.getElementById("timer").innerHTML = "Processing time: " + (elapsedTime / 1000).toFixed(2) + " seconds";
                }, 100);
            }
            function formatOutput(response) {
                // Format the response here as needed
                return response;
            }
        </script>
    </body>
    </html>
    '''

@app.route('/submit', methods=['POST'])
def submit():
    text = request.form['text']
    pdf_path = request.form['pdf_path']
    question = request.form['question']
    if pdf_path:
        try:
            pdf_text = convert_pdf_to_text(pdf_path)
            text = pdf_text if text else pdf_text + text
        except Exception as e:
            return str(e)
    start_time = time.time()
    try:
        vectorstore = create_vectorstore(text)
        response = rag_chain(vectorstore, text, question)
        end_time = time.time()
        processing_time = end_time - start_time
        return f"{response}<br><br>Processing time: {processing_time:.2f} seconds"
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)

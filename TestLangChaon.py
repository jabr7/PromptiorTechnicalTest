# Importación de las librerías necesarias
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Annoy  # Assuming Annoy is implemented in langchain_community.vectorstores
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from selenium import webdriver
from bs4 import BeautifulSoup

from selenium.webdriver.chrome.service import Service
import time

# Specify the path to chromedriver.exe
webdriver_path = './chromedriver.exe'

# Set the path using the Service class
service = Service(webdriver_path)
driver = webdriver.Chrome(service=service)

# Replace 'your_url_here' with the actual URL you want to scrape
url = "https://www.promptior.ai/about"

# Open the URL
driver.get(url)

# Wait for the page to load (adjust the sleep time based on your page's loading time)
time.sleep(1)

# Get the page source after waiting for a while to allow dynamic content to load
page_source = driver.page_source

# Parse the HTML content
soup = BeautifulSoup(page_source, 'html.parser')

# Extracting text from all paragraphs within the "text-section"
text_section = soup.find('div', class_='text-section')
if text_section:
    # Use find_all instead of find to get all paragraphs
    about_texts = text_section.find_all('p', class_='about-text')
    about_texts = ' '.join(p.get_text() for p in about_texts)  # Extract text and join into a single string
else:
    print("Text section not found")

# Close the browser
driver.quit()


# Initialization of Ollama with the Llama2 model
llm = Ollama(model="llama2")

# Creation of the embedding model
embeddings = OllamaEmbeddings()

# Indexing of the data
text_splitter = RecursiveCharacterTextSplitter()
docs = [Document(page_content=about_texts)]  # Pass about_texts as page_content
documents = text_splitter.split_documents(docs)
vector = Annoy.from_documents(documents, embeddings)
retriever = vector.as_retriever()


# First we need a prompt that we can pass into an LLM to generate this search query
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# Create a new chain to continue the conversation with these retrieved documents in mind
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


chat_history = []

while True:
    user_input = input("Enter your question: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    print(response["answer"])
    chat_history.append(AIMessage(content=response["answer"]))
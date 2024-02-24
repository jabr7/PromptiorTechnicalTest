from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Annoy 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
import time
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Set path to chromedriver as per your configuration
webdriver_path = ChromeDriverManager().install()

# Choose Chrome Browser
driver = webdriver.Chrome(service=Service(webdriver_path), options=chrome_options)

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



# Creation of the embedding model
embeddings = OllamaEmbeddings()

# Indexing of the data
text_splitter = RecursiveCharacterTextSplitter()
docs = [Document(page_content=about_texts)]
documents = text_splitter.split_documents(docs)
# I had some issues with FAISS for cpu on windows so I used Annoy instead
# Retriever is created using the Annoy vector store
vector = Annoy.from_documents(documents, embeddings)
retriever = vector.as_retriever()




# Initialization of Ollama with the Llama2 model
llm = Ollama(model="llama2")
# First we need a prompt that we can pass into an LLM to generate this search query
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
print(retriever_chain)

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
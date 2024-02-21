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

from selenium import webdriver
from bs4 import BeautifulSoup

from selenium.webdriver.chrome.service import Service
import time

# Specify the path to chromedriver.exe
webdriver_path = 'C:\chromedriver.exe'

# Set the path using the Service class
service = Service(webdriver_path)
driver = webdriver.Chrome(service=service)

# Replace 'your_url_here' with the actual URL you want to scrape
url = "https://www.promptior.ai/about"

# Open the URL
driver.get(url)

# Wait for the page to load (adjust the sleep time based on your page's loading time)
time.sleep(5)

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


# Creation of the retrieval chain
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Use the retrieval chain to ask a question with the Document as context
response = retrieval_chain.invoke({"input": "what is promptior?", "context": docs})
print(response["answer"])

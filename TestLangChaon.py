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

#Since it seems i can open a browser in railway i will be harcoding the text for the context window
# but the version with the browser was tested and its working
'''
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Set up headless Chrome
options = Options()
options.headless = True

# Use webdriver_manager to download and install the latest ChromeDriver
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

# Navigate to the page
driver.get('https://www.promptior.ai/about')

# Get the page content
page_content = driver.page_source
'''
page_content = """
<div class="about-page">
    <h2 class="section-title">Who are we?</h2>
    <div class="text-section">
        <p class="about-text aos-init aos-animate" data-aos="fade-left">Promptior was founded in March 2023 with the mission to democratize and facilitate access to Artificial Intelligence for people and organizations worldwide. The adoption of these technologies drives innovation, improves efficiency, and radically transforms the way people work, learn, and collaborate.</p>
        <p class="about-text aos-init aos-animate" data-aos="fade-right">The biggest challenge organizations will face in the coming years will be to fully leverage the potential that these technologies offer to their businesses and work teams. This is achieved through process redesign, integration with existing systems, and, above all, daily and collaborative adoption by organization members.</p>
        <p class="about-text aos-init" data-aos="fade-left">Our team is composed of professionals with extensive experience in technology and artificial intelligence, and we are committed to providing the tools, resources, and knowledge necessary to help our clients rapidly adopt these benefits.</p>
        <p class="about-text aos-init" data-aos="fade-right">We take pride in our dedication to our client's success and the advancement of AI worldwide. We are committed to building a smarter and more connected future for all.</p>
    </div>
</div>
"""

# Parse the HTML content
soup = BeautifulSoup(page_content, 'html.parser')

# Extracting text from all paragraphs within the "text-section"
text_section = soup.find('div', class_='text-section')
if text_section:
    # Use find_all instead of find to get all paragraphs
    about_texts = text_section.find_all('p', class_='about-text')
    about_texts = ' '.join(p.get_text() for p in about_texts)  # Extract text and join into a single string
else:
    print("Text section not found")


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
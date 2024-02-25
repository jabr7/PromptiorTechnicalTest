# Project Overview

This project is a solution to a challenge that involves creating an interactive chatbot that i resolved using the FastAPI framework for the interactions and the LangChain library for the bot, also using Llama as the ML model. The chatbot is capable of answering questions based on a predefined context, which in this case is the "About Us" section of the Promptior Website that had all the necessary information to answer the challenge questions.

## Implementation Logic

The project uses the LangChain library to create a retrieval chain that can generate responses based on the chat history and the input question. The retrieval chain consists of two parts: a retriever chain and a document chain. The retriever chain uses a language model to generate a search query based on the chat history, and the document chain uses the same language model to generate a response based on the retrieved documents and the chat history.

The FastAPI framework is used to create a REST API that allows users to interact with the chatbot. The API has a single endpoint (`/ask`) that accepts a question as input and returns the chatbot's response.

## Main Challenges and Solutions

One of the main challenges was to maintain the chat history between different API calls. This was solved by using a global variable to store the chat history. (This can be improved using a dictionary if we want to have different history ids and sessions)

Another challenge was to extract the context from a webpage. This was solved by using the BeautifulSoup library to parse the HTML content of the webpage and extract the text from the relevant sections. Originally i made this dinamic where it opened the chrome windows but after trying it in different deploy situations i couldnt managed to make it work headless so i hardcoded the text. The chrome version is commented and working if you launch it locally.

Finally, there were some issues with using the FAISS library for vector storage on Windows since i couldnt find a compatible version. This was solved by using the Annoy library instead.

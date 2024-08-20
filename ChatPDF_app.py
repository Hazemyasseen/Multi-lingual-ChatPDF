import os
import ChatPDF_utils
from tempfile import NamedTemporaryFile
from googletrans import Translator
import streamlit as st

# Set up the Streamlit app
st.title("Multilingual ChatPDF")
st.write("Upload a pdf file and ask any question. The bot will answer based on the content of the file.")

# File uploader widget to upload a PDF file
uploaded_file = st.file_uploader("Choose a pdf file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Load the model using a custom utility function
    MODEL_NAME = "qwen2:0.5b"
    model = ChatPDF_utils.load_model_Ollama(MODEL_NAME)
        
    # Embed the text from the PDF into a vector database
    retriever = ChatPDF_utils.embed_text_into_vector_db(temp_file_path)
        
    # Remove the temporary file after processing
    os.remove(temp_file_path)

    st.write("File uploaded and indexed successfully!")

    question = st.text_input("Ask a question:")
    if question:
        # Translate the question to English
        translator = Translator()
        r = translator.detect(question)
        question_in_english = translator.translate(question,dest ='en')

        # Create the RAG (Retrieval-Augmented Generation) chain
        rag_chain = ChatPDF_utils.create_prompt(model, retriever)

        # Get the answer from the model
        outputs = rag_chain.invoke(question_in_english.text)["text"]

        # Translate the answer back to the original language
        answer = translator.translate(outputs,dest = r.lang)

        # Display the answer
        st.write(answer.text)
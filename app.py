import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI


with st.sidebar:
    st.title("💬Chat with PDF📜")
    st.markdown('''
    ## The apps allow you to chat with your PDF.
    ##### 1. Upload your PDF
    ##### 2. Chat with your PDF
    ''')

    add_vertical_space(23)
    st.write('Made by Haseeb')


load_dotenv()
def main():
    st.header("Chat with PDF📜")

    add_vertical_space(3)
    #upload the pdf file
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    # st.write(uploaded_file.name)


    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        st.write("File Uploaded Successfully!!")

        #extracting the text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #splitting the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)


       
        # storing the vector and checking if the vector 
        store_name = uploaded_file.name[:-4]
        if os.path.exists(f"{store_name}.pkl"): #checking if the file path exist of the embeddings
            with open(f"{store_name}.pkl","rb") as f:
                vectorstores = pickle.load(f)
            # st.write("Embedding loaded from the disk")
        else:
            #embedding the text
            embeddings = OpenAIEmbeddings()
            # embedding = embeddings.embed_text(text=chunks)
            vectorstores = FAISS.from_texts(chunks, embedding= embeddings)
        

            with open(f"{store_name}.pkl","wb") as f:  #if the path dont exists saving the embeddings
                pickle.dump(vectorstores, f)

            # st.write("Embedding is done")
        add_vertical_space(12)
        #chatbot
        query = st.text_input("Enter the text:", placeholder= "Ask question related to PDF...")


        # st.write(query)

        if query :
            doc = vectorstores.similarity_search(query = query, k=2)
            llm = ChatOpenAI(model_name = 'gpt-3.5-turbo')   
            chain = load_qa_chain(llm = llm, chain_type="stuff")

            with get_openai_callback() as cb: #checking the cost of each query
                response = chain.run(input_documents =  doc, question = query)
                print(cb)
            st.write(response)
        


    else:
        st.write("Please upload a PDF file")





if __name__ == "__main__":
    main()



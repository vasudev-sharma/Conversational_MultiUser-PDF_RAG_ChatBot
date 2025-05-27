#import Essential dependencies
import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

try:
        load_dotenv(".env")
except Exception as e:
        raise Exception("Please create a .env file with your OpenAI API key") from e


#function to load the vectordatabase
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=os.environ.get("OPENAI_API_KEY"))
        return llm

#creating prompt template using langchain
def load_prompt():
        prompt = """ You're a PDF chatbot helping the users to guide to the answer as relevant as possible based on a PDF. You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}

        Rule:   
        Use the following rules -      
                - if the answer is not in the pdf answer, the respond with: "I don't know."
                - Only answer based on the provided information. Do not make up information.
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt


def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


if __name__=='__main__':
        sl.header("welcome to the 📝PDF bot")
        sl.write("🤖 You can chat by Entering your queries ")
        knowledgeBase=load_knowledgeBase()
        llm=load_llm()
        prompt=load_prompt()
        
        query=sl.text_input('Enter some text')
        

        # TODO: Experiment with Query Expansion
        
        if(query):
                #getting only the chunks that are similar to the query for llm to produce the output
                
                # Not to build
                retriever = knowledgeBase.as_retriever()

                # TODO: Remove me as we are calling it twice
                # docs = retriever.get_relevant_documents(query)

                # Load the vector store only once and use it for similarity search for query
                # similar_embeddings=knowledgeBase.similarity_search(query)
                # similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY")))
                
                #creating the chain for integrating llm,prompt,stroutputparser
                # retriever = similar_embeddings.as_retriever()

                # TODO: Experiment with Hybrid retreiver (keyword + search)
                rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                
                response=rag_chain.invoke(query)
                sl.write(response)
        
                
        
        
        
        
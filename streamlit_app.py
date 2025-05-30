# import Essential dependencies
import os
import uuid
from typing import List

import pandas as pd
import streamlit as sl
import streamlit as st
from datasets import Dataset
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import \
    StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from prometheus_client import REGISTRY, Counter, Histogram, start_http_server
from pydantic import Field
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from streamlit_extras.prometheus import streamlit_registry
import time
from utils import get_chat_history, insert_application_logs


def get_llm_queries_counter():
        """
        Creates and returns a Counter metric for tracking the total number of LLM queries.
        Returns:
                Counter: A Prometheus Counter object named "llm_queries_total" with documentation "Total LLM Queries",
                                registered to the Streamlit metrics registry.
        """
    
        return Counter(
                name="llm_queries_total",
                documentation="Total LLM Queries",
                registry=streamlit_registry(),
        )


@st.cache_resource
def get_llm_errors_counter():
        """
        Creates and returns a Counter metric for tracking the total number of LLM errors.
        Returns:
                Counter: A Prometheus Counter metric named 'llm_errors_total' with documentation
                'Total LLM Errors', registered to the Streamlit metrics registry.
        """
        
        return Counter(
        name="llm_errors_total",
        documentation="Total LLM Errors",
        registry=streamlit_registry(),
        )


@st.cache_resource
def get_llm_latency_histogram():
        """
        Creates and returns a Prometheus Histogram metric for tracking LLM query latency in seconds.
        Returns:
                Histogram: A Prometheus Histogram object configured to record LLM query latency.
        Raises:
                Any exception raised by the Histogram constructor or streamlit_registry().
        Example:
                histogram = get_llm_latency_histogram()
                histogram.observe(1.23)
        """
        
        return Histogram(
        name="llm_query_latency_seconds",
        documentation="LLM Query latency (seconds)",
        registry=streamlit_registry(),
        )



try:
    load_dotenv(".env")
except Exception as e:
    raise Exception("Please create a .env file with your OpenAI API key") from e


# function to load the vectordatabase
def load_knowledgeBase():
        """
        Loads a FAISS-based knowledge base from a local vector store using OpenAI embeddings.
        Returns:
                FAISS: An instance of the FAISS vector store loaded with the specified embeddings.
        Raises:
                ValueError: If the OpenAI API key is not set in the environment variables.
                Exception: If loading the FAISS vector store fails.
        """
    
        embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
        DB_FAISS_PATH = "vectorstore/db_faiss"
        db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
        return db


# function to load the OPENAI LLM
def load_llm():
        """
        Initializes and returns a ChatOpenAI language model instance using the OpenAI API key from environment variables.
        Returns:
                ChatOpenAI: An instance of the ChatOpenAI language model with the specified configuration.
        Raises:
                KeyError: If the 'OPENAI_API_KEY' environment variable is not set.
        """
        
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
        )
        return llm


# creating prompt template using langchain
def load_prompt():
        """
        Loads and returns a chat prompt template for a PDF-based chatbot.
        The prompt instructs the chatbot to answer user questions strictly based on the content of a provided PDF.
        It includes placeholders for chat history, context, and question, and enforces rules such as:
                - Responding with "I don't know. Please ask questions relevant to the PDF document" if the answer is not found in the PDF.
                - Only answering based on the provided information without making up content.
        Returns:
                ChatPromptTemplate: A prompt template configured with the specified instructions and placeholders.
        """
        prompt = """ You're a PDF chatbot helping the users to guide to the answer as relevant as possible based on a PDF. You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        Current conversation:
        {chat_history}

        context = {context}
        question = {question}

        Rule:   
        Use the following rules -      
                - if the answer is not in the pdf answer, the respond with: "I don't know. Please ask questions relevant to the PDF document"
                - Only answer based on the provided information. Do not make up information.
                """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query_llm(query, rag_chain):
        """
        Executes a query against a provided RAG (Retrieval-Augmented Generation) chain and tracks metrics.
        This function increments counters for LLM queries and errors, measures latency, and observes latency metrics.
        It invokes the RAG chain with the given query and returns the response. If an exception occurs during invocation,
        the error counter is incremented and the exception is propagated.
        Args:
                query (str): The user query to be processed by the RAG chain.
                rag_chain: The RAG chain object with an `invoke` method that processes the query.
        Returns:
                The response from the RAG chain's `invoke` method.
        Raises:
                Exception: Propagates any exception raised during the RAG chain invocation.
        """
        
        LLM_QUERIES = get_llm_queries_counter()
        LLM_ERRORS = get_llm_errors_counter()
        LLM_LATENCY = get_llm_latency_histogram()
        LLM_QUERIES.inc()
        start_time = time.time()

        try:
        latency = time.time() - start_time
        response = rag_chain.invoke({"question": query})
        LLM_LATENCY.observe(latency)
        return response
        except Exception as e:
        LLM_ERRORS.inc()
        raise


# --- Keyword Retriever (Simple Implementation) ---

class KeywordRetriever(BaseRetriever):
        """
        A retriever class that selects the top-k documents most relevant to a query based on keyword frequency.
        Attributes:
                documents (List[Document]): The list of documents to search through.
                k (int): The number of top relevant documents to return.
        Methods:
                _get_relevant_documents(query, run_manager=None):
                        Returns a list of up to k documents that contain the highest frequency of query terms.
                        The relevance score is computed as the sum of occurrences of each query term in the document content.
                        Only documents with a positive score are returned.
                _aget_relevant_documents(query, run_manager=None):
                        Asynchronous version of _get_relevant_documents.
        """

        documents: List[Document] = Field(default_factory=list)
        k: int = 5

        def _get_relevant_documents(self, query, run_manager=None):
                query_terms = set(query.lower().split())
                scored = []
                for doc in self.documents:
                        content = doc.page_content.lower()
                        score = sum(content.count(term) for term in query_terms)
                        scored.append((score, doc))
                scored.sort(reverse=True, key=lambda x: x[0])
                return [doc for score, doc in scored[: self.k] if score > 0]

        async def _aget_relevant_documents(self, query, run_manager=None):
                return self._get_relevant_documents(query, run_manager)




def run_evaluation(rag_chain):
        """
        Evaluates a Retrieval-Augmented Generation (RAG) chain using mock evaluation data.
        This function generates evaluation results by invoking the provided RAG chain on a set of predefined
        question-answer pairs. It collects the generated answers, compares them with ground truth answers,
        and computes evaluation metrics such as faithfulness, answer relevancy, and context precision.
        Args:
                rag_chain: An object representing the RAG chain, which must implement an `invoke` method that
                                accepts a dictionary with a "question" key and returns a dictionary with an "answer" key.
        Returns:
                dict: The evaluation results containing computed metrics for the RAG chain's responses.
        """
        

        # TODO: LLM as a Judge

        # Mock evaluation data

        evaluation_data = [
        {
                "question": "Who is Arthur Samuel?",
                "ground_truth": "Arthur Samuel was a pioneer in machine learning.",
                "contexts": [
                "Arthur Samuel was one of the pioneers of machine learning and artificial intelligence."
                ],
        },
        # Add more QA pairs as needed
        ]

        results_eval = []
        for item in evaluation_data:
        query = item["question"]
        response = rag_chain.invoke({"question": query})
        generated_answer = response["answer"]
        results_eval.append(
                {
                "question": query,
                "ground_truth": item["ground_truth"],
                "retrieved_contexts": [],  # add more context here,
                "response": generated_answer,
                }
        )

        df = pd.DataFrame(results_eval)
        ragas_dataset = Dataset.from_pandas(df)
        metrics = [faithfulness, answer_relevancy, context_precision]

        results_eval = evaluate(ragas_dataset, metrics)
        return results_eval


if __name__ == "__main__":

    sl.header("welcome to the 📝PDF bot")
    sl.write("🤖 You can chat by Entering your queries ")

    # Multi-user chatbot
    if "session_id" not in sl.session_state:
        sl.session_state["session_id"] = str(uuid.uuid4())
    session_id = sl.session_state["session_id"]

    knowledgeBase = load_knowledgeBase()
    history = StreamlitChatMessageHistory(key="chat_history")

    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=history, return_messages=True
    )
    llm = load_llm()
    prompt = load_prompt()
    query = sl.text_input("Enter some text")

    chat_history = get_chat_history(session_id)

    # TODO: Experiment with Query Expansion

    if query:
        # getting only the chunks that are similar to the query for llm to produce the output

        # Not to build
        # TODO: Finetune retriever
        all_docs = knowledgeBase.similarity_search("", k=1000)

        vector_retriever = knowledgeBase.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        # keyword_retriever = SimpleKeywordRetriever(all_docs, k=5)

        # --- Hybrid Retrieval: Combine results from both retrievers ---

        keyword_retriever = KeywordRetriever(documents=all_docs, k=5)

        # --- Use EnsembleRetriever for hybrid search ---
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.5, 0.5],  # You can tune the weights
        )



        # TODO: Add memory to RAG
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

        response = query_llm(query, rag_chain)
        sl.write(response["answer"])

        # Insert into DB
        insert_application_logs(
            session_id, query, response["answer"], model="gpt-3.5-turbo"
        )


        print("\n\n")
        print("*********" * 10)

        # Strealit UI
        sl.title("RAG Evaluation")
        if sl.button("Run RAG Evaluation"):

            results = run_evaluation(rag_chain)
            print(results)

            sl.success("Evaluation complete!")
            sl.write("**RAG Evaluation Results:**")
            sl.write(results.to_pandas())

            with open("rag_evaluation_log.txt", "a") as f:
                f.write(str(results) + "\n")
            sl.info("Results logged to rag_evaluation_log.txt")

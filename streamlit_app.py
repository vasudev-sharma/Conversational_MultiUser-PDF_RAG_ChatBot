# import Essential dependencies
import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from typing import List
from langchain_core.documents import Document
from pydantic import Field
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from ragas.metrics import faithfulness, answer_relevancy, context_precision
import os
import threading
from utils import LoggingRetriever
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
import pandas as pd
from datasets import Dataset
from langsmith import Client
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from streamlit_extras.prometheus import streamlit_registry
import streamlit as st

from prometheus_client import start_http_server, Counter, Histogram, REGISTRY


def get_llm_queries_counter():
    return Counter(
        name="llm_queries_total",
        documentation="Total LLM Queries",
        registry=streamlit_registry(),
    )


@st.cache_resource
def get_llm_errors_counter():
    return Counter(
        name="llm_errors_total",
        documentation="Total LLM Errors",
        registry=streamlit_registry(),
    )


@st.cache_resource
def get_llm_latency_histogram():
    return Histogram(
        name="llm_query_latency_seconds",
        documentation="LLM Query latency (seconds)",
        registry=streamlit_registry(),
    )


import time


# def start_prometheus_server(): # run in background thread
#         start_http_server(9090)

# threading.Thread(target=start_prometheus_server, daemon=True).start()


try:
    load_dotenv(".env")
except Exception as e:
    raise Exception("Please create a .env file with your OpenAI API key") from e


# function to load the vectordatabase
def load_knowledgeBase():
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return db


# function to load the OPENAI LLM
def load_llm():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return llm


# creating prompt template using langchain
def load_prompt():
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


# To make SimpleKeywordRetriever compatible, subclass BaseRetriever:
class KeywordRetriever(BaseRetriever):
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


from langchain_core.messages import HumanMessage, AIMessage
import uuid
from utils import get_chat_history, insert_application_logs


def run_evaluation(rag_chain):

    # TODO use LLM to generation data

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

        # retriever = LoggingRetriever(retrieval=)

        # TODO: Remove me as we are calling it twice
        # docs = retriever.get_relevant_documents(query)

        # Load the vector store only once and use it for similarity search for query
        # similar_embeddings=knowledgeBase.similarity_search(query)
        # similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY")))

        # creating the chain for integrating llm,prompt,stroutputparser
        # retriever = similar_embeddings.as_retriever()

        # TODO: Experiment with Hybrid retreiver (keyword + search)
        # rag_chain = (
        #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
        #         | prompt
        #         | llm
        #     )

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

        # # converasation_rag
        # chat_history = sl.session_state.chat_history

        # # Handling
        # chat_history.extend([HumanMessage(query), AIMessage(response['answer'])])
        # print(chat_history)

        # # Contextualize system prompts
        # contextualize_q_system_prompt = """
        # Given a chat history and the latent user question
        # which might reference context in the chat history,
        # formulate a standalone question which can be understood
        # without the chat history. DO NOT answer the question,
        # just reformulate it if needed and otherwise return it as is.
        # """

        # contextualize_q_system_prompt = ChatPromptTemplate.from_messages(

        #        [
        #               ("system", contextualize_q_system_prompt),
        #               MessagesPlaceholder("chat_history"),
        #               ("human", "{input}")
        #        ]
        # )

        # print("\n"*2)
        # print("*********"*10)
        # print("The chat history is \n: ", chat_history)
        # print("\n\n")

        # Testing prompts
        # Who is Arthur Samuel?
        # Where he was working in 1959?
        # contextualize_chain = contextualize_q_system_prompt | llm | StrOutputParser()
        # context_reponse = contextualize_q_system_prompt.invoke({"input": query, "chat_history": chat_history})
        # print("\n"*2)
        # print("*********"*10)
        # print(context_reponse)
        # # print(context_reponse.content)
        # print("\n"*2)

        # Update the streamlit variable
        # sl.session_state.chat_history = chat_history

        # TODO: Evaluate responses

        # evaluation data: TODO (Add Eval / question pairs or use LLM)

        # faithfulness_score = []
        # answer_relevancy_score = []
        # context_precision_score = []

        # for result in results:
        #         faith = faithfulness(result['generated_answer'], result['context'], result['ground_truth'])
        #         relevancy = answer_relevancy(result["generated_answer"], result["ground_truth"])
        #         precision = context_precision(result["context"], result["ground_truth"])

        #         faithfulness_score.append(faith)
        #         answer_relevancy_score.append(relevancy)
        #         answer_relevancy_score.append(precision)

        # print("\n\n")
        # print("*********"*10)
        # # aggregate results
        # print("Faithfulness: ", sum(faithfulness_score) / len(faithfulness_score))
        # print("Answer Relevancy: ", sum(answer_relevancy_score) / len(answer_relevancy_score))
        # print("Context Precision: ", sum(context_precision_score) / len(context_precision_score))

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

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:22:00 2025

@author: user
"""

import pandas as pd
import numpy as np
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import OpenAI, LLMChain, PromptTemplate
from collections import Counter


def tokenize(text):
    return text.lower().split()



# Find the document with the most common words
def find_best_match(query, documents):
    query_tokens = Counter(tokenize(query))
    best_match = None
    max_overlap = 0

    for doc in documents:
        doc_tokens = Counter(tokenize(doc))
        overlap = sum((query_tokens & doc_tokens).values())
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = doc

    return best_match


   

def query_bot_with_llm(q,seg_clien_text,seg_item_text,chain):
    # Retrieve data
    #customer_results = docsearch_client_category.similarity_search(q, k=1)
    #customer_info = customer_results[0].page_content if customer_results else "No relevant customer data found."
    
    customer_info=find_best_match(q, seg_clien_text)

    #item_results =docsearch_item_category.similarity_search(q, k=1)
    #item_info = item_results[0].page_content if item_results else "No relevant item data found."
    item_info= find_best_match(q, seg_item_text)
    # Apply decision logic
  

    # Generate natural response with LLM
    response = chain.run({
        "customer_data": customer_info,
        "item_data": item_info,
        "question": q
    })

    return response



# Tokenize query and documents




class decision_bot():
    
        #Get the API for LLM
 
        
    groq_api = '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'
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)
    
    #Define the embedding used
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    #embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    
    
    #Get the segmentation of the client
    
    seg_clien=pd.read_excel('client_segmentation.xlsx')
    seg_clien=seg_clien[['Customer', 'Customer Segment']]
    # Remove the word "Customer"
    seg_clien['Customer Segment'] = seg_clien['Customer Segment'].str.replace('Customer', '', regex=False).str.strip()
    #Convert to text data
    seg_clien_text = seg_clien.apply(lambda row: f"Customer: {row['Customer']}, Category: {row['Customer Segment']}", axis=1).tolist()
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    #text_chunks = text_splitter.create_documents(seg_clien_text)
    #docsearch_client_category = FAISS.from_documents(text_chunks, embeddings)


    #Get the segment of the Item
    seg_item=pd.read_excel('product_classification.xlsx')
    seg_item=seg_item[['Item', 'Category',  'XYZ Category']]
    seg_item.columns=['Item','ABC Category','XYZ Category']
    seg_item['Category']=seg_item['ABC Category']+ seg_item['XYZ Category']
    seg_item=seg_item[['Item','Category']]
    #Convert to text data
    seg_item_text = seg_item.apply(lambda row: f"Item: {row['Item']}, Category: {row['Category']}", axis=1).tolist()
    #Split into chunks
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    #text_chunks = text_splitter.create_documents(seg_item_text)
    #docsearch_item_category = FAISS.from_documents(text_chunks, embeddings)

    prompt_template = """
    You are a decision bot that helps a retailer decide whether to order an item for a customer or
    not based on the segment of the customer and the category of the item.
    You will be provided a dataset that specifies the segment of the customer and a dataset that shows the category of the item.
    Based on the datasets and rules provided, answer the user's query:
    Rules:
    - If the segment of the customer is Top, then order whatever item regardless of the category of the item.
    - If the segment of the customer is High Value, then order items that don't have the category CZ.
    - If the segment of the customer is Medium Value, then order items that don't have the following category CY, CZ.
    - If the segment of the customer is Low Value, then order items that don't have the following category CX, CY, CZ, and BZ.
    - If the segment of the customer is Lost, then order items that have the category AX and AY.
    
    When answering the question, don't go into detailed explanation just give a brief explanation. 
    Ask a followup question when done from the decision.
    
    Question: {question}
    
    Customer Data: {customer_data}
    
    Item Data: {item_data}
    
    
    """

    prompt = PromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)




















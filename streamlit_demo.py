# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 23:06:37 2025

@author: user
"""

import streamlit as st

from decision_bot import tokenize, find_best_match,query_bot_with_llm
from decision_bot import decision_bot


bot = decision_bot()


st.set_page_config(page_title="Decision Maker Bot")


with st.sidebar:
    st.title('Decision Maker Bot')
    
    
    
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello, I am here to help you with decision making"}]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)
        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Looking into your customer and items data"):
            seg_clien_text=bot.seg_clien_text 
            seg_item_text=bot.seg_clien_text
            chain=bot.chain
            response=query_bot_with_llm(input,seg_clien_text,seg_item_text,chain) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
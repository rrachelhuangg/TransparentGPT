import os
import chainlit as cl
from langchain.memory.buffer import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains import LLMChain
from prompts import default_prompt_template, doctor_prompt_template, default_prompt_template_no_sources, doctor_prompt_template_no_sources
from dotenv import load_dotenv
from chainlit.input_widget import Select, Switch, Slider
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from math import exp
import numpy as np
from typing import Any, Dict, List, Tuple
from langchain_core.output_parsers import BaseOutputParser
from difflib import SequenceMatcher

llm = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    model =  "meta-llama/Llama-3.3-70B-Instruct",
    temperature = 0.7
).bind(logprobs=True)

def direct_text_comparison(text1, text2):
    similarity_ratio = SequenceMatcher(None, text1, text2).ratio()
    return similarity_ratio

def context_text_comparison(text1, text2):
    similarity_ratio = llm.predict(f"How similar in context and meaning are these two pieces or provided text? Please provide a two-sentence analysis. End your response with either not similar, kind of similar, and very similar. Text1: {text1} Text 2: {text2}")
    return similarity_ratio

def source_description(source):
    description = llm.predict(f"Please provide a two-sentence description of the information that this source contains and where it got its information from: {source}")
    return description

def similarity_value(direct, context):
    context_similarity, direct_similarity = 0, 0
    context = context[-50:]
    if "not similar" in context:
        context_similarity = 0.25
    elif "kind of similar" in context:
        context_similarity = 0.5
    elif "very similar" in context:
        context_similarity = 0.75
    if direct < 0.25:
        direct_similarity = 0.25
    elif direct > 0.25 and direct < 0.5:
        direct_similarity = 0.5
    elif direct > 0.5:
        direct_similarity = 0.75
    print("VALS: ", context_similarity, direct_similarity)
    return context_similarity + direct_similarity

def test_sim_val(response_content, link):
    text2 = source_description(link)
    #context_text_comp = context_text_comparison(response_content, text2)
    return direct_text_comparison(response_content, link)
    # context = context_text_comp[-50:]
    # if "not similar" in context:
    #     context_similarity = 0.25
    # elif "kind of similar" in context:
    #     context_similarity = 0.5
    # elif "very similar" in context:
    #     context_similarity = 0.75
    # return context_similarity

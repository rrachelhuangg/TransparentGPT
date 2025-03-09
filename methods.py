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
import requests
from bs4 import BeautifulSoup
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

llm = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    model =  "meta-llama/Llama-3.3-70B-Instruct",
    temperature = 0.7
).bind(logprobs=True)

def get_wikipedia_page_content(page_title):
    #scraping wikipedia pages with the Revisions API
    page_title = re.sub(r"\s+", "", page_title).strip()
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&titles={page_title}&formatversion=2&rvprop=content&rvslots=*"
    response = requests.get(url)
    data = response.json()
    return data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]

def test_scrape_sim(link, response):
    tfidf_vectorizer = TfidfVectorizer()
    try:
        idx = link.rfind("/")
        title = link[idx+1:]
        tfidf_matrix = tfidf_vectorizer.fit_transform([get_wikipedia_page_content(title), response])
        # tfidf_matrix = tfidf_vectorizer.fit_transform([scrape_web_text(link), response])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return cosine_sim*100
    except:
        return 0

config_file="config.json"
def get_config():
    with open(config_file, "r") as file:
        return json.load(file)
def update_config(new_value):
    config = get_config()
    config["num_sources"] = new_value
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)

def load_config():
    with open("config.json","r") as file:
        return json.load(file)

def generate_hypothetical_answer(question: str) -> str:
    """Have LLM generate a hypothetical answer to assist with bot response."""
    prompt = PromptTemplate(
        input_variables=['question'],
        template="""
        You are an AI assistant taked with generate a hypothetical answer to the following question. Your answer shoulld be detailed and comprehensive,
        as if you had access to all relevant information. This hypothetical answer will be used to improve document retrieval, so include key terms and concepts
        that might be relevant. Do not include phrases like "I think" or "It's possible that" - present the information as if it were factual.
        Question:{question}
        Hypothetical answer:
        """,
    )
    return TransparentGPT_settings.llm.invoke(prompt.format(question=question))

def highest_log_prob(vals):
    """Calculates the perplexity score (confidence) of bot response."""
    logprobs = []
    for token in vals:
        logprobs += [token['logprob']]
    average_log_prob = sum(logprobs)/len(logprobs)
    return np.round(np.exp(average_log_prob)*100,2)

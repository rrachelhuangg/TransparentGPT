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

def scrape_web_text(link):
    page = requests.get(link)
    soup = BeautifulSoup(page.content, "html.parser")
    try:
        text = soup.get_text()
        sentences = text.split(".")
        results = [sentence for sentence in sentences if "Instagram" in sentence]
        final = ""
        for result in results:
            final += re.sub(r"\s+", "", text)
        return final
    except:
        return ""

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
#update_config(5)

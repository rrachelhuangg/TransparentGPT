"""
Methods to assist TransparentGPT bulk logic.
"""
import os
import json
import re
import requests
import chainlit as cl
from math import exp
import numpy as np
from prompts import default_prompt_template, doctor_prompt_template, default_prompt_template_no_sources, doctor_prompt_template_no_sources
from typing import List
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

llm = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    model =  "meta-llama/Llama-3.3-70B-Instruct",
    temperature = 0.7
).bind(logprobs=True)

def get_wikipedia_page_content(page_title):
    """
    Scrapes Wikipedia pages with the Revisions API and returns the main text content from the page.
    """
    page_title = re.sub(r"\s+", "", page_title).strip()
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&titles={page_title}&formatversion=2&rvprop=content&rvslots=*"
    response = requests.get(url)
    data = response.json()
    return data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]

def similarity_analysis(link, response):
    """
    Returns a percentage that represents how similar the Wikipedia page content and TransparentGPT response are.
    """
    tfidf_vectorizer = TfidfVectorizer()
    try:
        idx = link.rfind("/")
        title = link[idx+1:]
        tfidf_matrix = tfidf_vectorizer.fit_transform([get_wikipedia_page_content(title), response])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return cosine_sim*100
    except:
        return 0

config_file="config.json"
def get_config():
    """
    Get num_sources config.
    """
    with open(config_file, "r") as file:
        return json.load(file)
def update_config(new_value):
    """
    Update num_sources config.
    """
    config = get_config()
    config["num_sources"] = new_value
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)

def load_config():
    """
    Load num_sources config.
    """
    with open("config.json","r") as file:
        return json.load(file)

def highest_log_prob(vals):
    """Calculates the perplexity score (confidence) of bot response."""
    logprobs = []
    for token in vals:
        logprobs += [token['logprob']]
    average_log_prob = sum(logprobs)/len(logprobs)
    return np.round(np.exp(average_log_prob)*100,2)

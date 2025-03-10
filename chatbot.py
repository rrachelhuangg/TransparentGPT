import os
import chainlit as cl
from langchain.memory.buffer import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains import LLMChain
from prompts import default_prompt_template, doctor_prompt_template, default_prompt_template_no_sources, doctor_prompt_template_no_sources, default_quirky_genz_prompt, default_quirky_genz_prompt_no_sources, default_food_critic_prompt, default_food_critic_prompt_no_sources, default_media_critic_prompt, default_media_critic_prompt_no_sources
from dotenv import load_dotenv
from chainlit.input_widget import Select, Switch, Slider
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from math import exp
import numpy as np
from typing import Any, Dict, List, Tuple
from langchain_core.output_parsers import BaseOutputParser
from difflib import SequenceMatcher
from methods import test_scrape_sim, update_config, load_config, generate_hypothetical_answer, highest_log_prob
import json
from classes import LineListOutputParser, TransparentGPTSettings
import emoji

#setting environment variables (non-Nebius API access keys)
#HAVE CLASSES BE IMPORT FROM OTHER FILES TO CLEAN UP CODE!! proper documentation and typing are v important
#also don't forget to refactor!^
#have a requirements.txt file if not deployed?
load_dotenv()

config = load_config()
num_sources = config["num_sources"]

TransparentGPT_settings = TransparentGPTSettings()

@cl.on_chat_start
async def start():
    greeting = f"Hello! I am TransparentGPT, a chatbot that is able to clarify my reasoning 🧠, explain my thought process 🙊, and cite the sources 📚 that I used for my response. \n\n I also provide a suite of customizable features! 😁 \n\n You can find my customization options in the settings panel that opens up when you click on the gear icon below 🔨. \n\n Click on the ReadME button in the top right of your screen to learn more about how I work. 🫶"
    await cl.Message(greeting).send()
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Select Model",
                description="Choose which large language model you want to interact with.",
                values=["Meta Llama 3.1", "Meta Llama 3.3", "MistralAI", "Dolphin Mixtral", "Microsoft Mini"],
                initial_index=1,
            ),
            Switch(
                id="Display Sources",
                label="Display Sources",
                description = "Choose to have sources for response displayed.",
                initial=True
            ),
            Select(
                id="Prompt Template",
                label="Prompt Template",
                description="Determines the type of bot you interact with.",
                values=["default", "doctor", "genz", "food_critic", "media_critic"],
                initial_index=0,
            ),
            Select(
                id="Query Expansion",
                label="Use Query Expansion",
                description = "Use query expansion to improve response context.",
                items = TransparentGPT_settings.query_expansion_options,
                initial_value="No query expansion"
            ),
            Slider(
                id="Number of Sources",
                label="Number of Sources",
                description="Choose the number of sources you want the bot to use for its response.",
                initial=3,
                min=1,
                max=10,
                step=1
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                description="Choose the desired consistency of bot response.",
                initial=0.7,
                min=0,
                max=2,
                step=0.1
            ),
        ]
    ).send()

@cl.on_settings_update
async def start(settings):
    update_config(settings['Number of Sources'])
    TransparentGPT_settings.update_settings(settings)

@cl.on_message
async def handle_message(message: cl.Message):
    await cl.Message("Your message was received successfully. I am working on generating my response. Please wait for a few seconds...").send()
    question = message.content
    expanded_query = ''
    if TransparentGPT_settings.query_expansion != 'No query expansion':
        if TransparentGPT_settings.query_expansion == 'Basic query expansion':
            t = 'Return a one sentence thorough description of this content: {question}'
            pt = PromptTemplate(input_variables=['question'], template=t)
            init_chain = pt | TransparentGPT_settings.llm
            expanded_query = init_chain.invoke({"question": message.content, "num_sources": TransparentGPT_settings.num_sources}).content
        elif TransparentGPT_settings.query_expansion == 'Multiquery expansion':
            output_parser = LineListOutputParser()
            pt = PromptTemplate(
                input_variables=['question'],
                template="""
                You are an AI language model assistant. Your task is to generate give different versions of the given user question to retrieve
                context for your response. By generating multiple perspectives on the user question, your goal is to help the user overcome
                some of hte limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. 
                Original question: {question},
                """
            )
            init_chain = pt | TransparentGPT_settings.llm | output_parser
            expanded_query = ' '.join(init_chain.invoke({'question': message.content, "num_sources": TransparentGPT_settings.num_sources}))
        elif TransparentGPT_settings.query_expansion == "Hypothetical answer":
            hypothetical_answer = generate_hypothetical_answer(message.content)
            expanded_query = f'{message.content} {hypothetical_answer.content}'
    if expanded_query!='':
        await cl.Message(f"Using {TransparentGPT_settings.query_expansion}, your query is now: {expanded_query}. This expanded query will help me find more relevant information for my response.").send()
    no_source_prompt=""
    if expanded_query == '' and not TransparentGPT_settings.display_sources:
        no_source_prompt = TransparentGPT_settings.prompt_mappings[TransparentGPT_settings.prompt_name+"_no_sources"]
        expanded_query = no_source_prompt.invoke({"question": question, "num_sources": TransparentGPT_settings.num_sources})
    elif expanded_query == '' and TransparentGPT_settings.display_sources:
        expanded_query = TransparentGPT_settings.prompt.invoke({"question":question, "num_sources":TransparentGPT_settings.num_sources})
    elif expanded_query !='' and not TransparentGPT_settings.display_sources:
        no_source_prompt = TransparentGPT_settings.prompt_mappings[TransparentGPT_settings.prompt_name+"_no_sources"]
        expanded_query = no_source_prompt.invoke({"question": expanded_query, "num_sources": TransparentGPT_settings.num_sources})
    elif expanded_query !='' and TransparentGPT_settings.display_sources:
        expanded_query = TransparentGPT_settings.prompt.invoke({"question":expanded_query, "num_sources":TransparentGPT_settings.num_sources})
    response = TransparentGPT_settings.llm.invoke(expanded_query)
    similarity_values = []
    await cl.Message("I have begun looking for relevant sources to answer your query, and am giving them a similarity score to show you how relevant they are to my response.").send()
    if no_source_prompt=="":
        temp = response.content
        sources = []
        count = 0
        while "*" in temp:
            if count < num_sources:
                link_idx = temp.rfind("*")
                source = temp[link_idx+1:]
                similarity_values += [test_scrape_sim(source, response.content)]
                temp = temp[:link_idx]
                count += 1
            else:
                break
    temp = response.content
    here = 2
    count = 0
    if len(similarity_values) > 0:
        while "*" in temp:
            if count < num_sources:
                link_idx = temp.rfind("*")
                response.content = response.content[:link_idx] + str(round(similarity_values[here],3)) + "%" + response.content[link_idx+1:]
                temp = temp[:link_idx]
                count += 1
                here -= 1
            else:
                break
    output_message = response.content + f"\n I am {highest_log_prob(response.response_metadata["logprobs"]['content'])}% confident in this response."
    await cl.Message(output_message).send()


if __name__ == '__main__':
    start()
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
from methods import test_scrape_sim

#setting environment variables (non-Nebius API access keys)
#HAVE CLASSES BE IMPORT FROM OTHER FILES TO CLEAN UP CODE!! proper documentation and typing are v important
#also don't forget to refactor!^
#have a requirements.txt file if not deployed?
load_dotenv()

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser that splits a LLM result into a list of queries."""
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split('\n')
        return list(filter(None, lines))

class TransparentGPTSettings:
    def __init__(self):
        self.model = "meta-llama/Llama-3.3-70B-Instruct"
        self.temperature = 0.7
        self.prompt = default_prompt_template
        self.prompt_mappings = {"default": default_prompt_template, "default_no_sources": default_prompt_template_no_sources, "doctor": doctor_prompt_template, "doctor_no_sources": default_prompt_template_no_sources}
        self.prompt_name = "default"
        self.llm = ChatOpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
            model = self.model,
            temperature = self.temperature
        ).bind(logprobs=True)
        self.display_sources = True
        self.query_expansion_options = {
            'No query expansion': 'No query expansion',
            'Basic query expansion': 'Basic query expansion',
            'Multiquery expansion': 'Multiquery expansion',
            'Hypothetical answer expansion': 'Hypothetical answer'
        }
        self.query_expansion = 'No query expansion'

    def update_settings(self, settings):
        self.model = settings['Model']
        self.temperature = settings['Temperature']
        self.prompt = self.prompt_mappings[settings['Prompt Template']]
        self.llm = ChatOpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
            model = self.model,
            temperature = self.temperature
        ).bind(logprobs=True)
        self.display_sources = settings['Source Display']
        self.prompt_name = settings['Prompt Template']
        self.query_expansion = settings['Query Expansion']

TransparentGPT_settings = TransparentGPTSettings()

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

@cl.on_chat_start
async def start():
    await cl.Message("Hello!").send()
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=50,
                                                   return_messages = True)
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1", "cognitivecomputations/dolphin-2.9.2-mixtral-8x22b", "Qwen/Qwen2.5-Coder-7B", "microsoft/Phi-3-mini-4k-instruct"],
                initial_index=1,
            ),
            Switch(id="Source Display", label="Display Sources", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=0.7,
                min=0,
                max=2,
                step=0.1
            ),
            Select(
                id="Prompt Template",
                label="Prompt used to create bot",
                values=["default", "doctor"],
                initial_index=0,
            ),
            Select(
                id="Query Expansion",
                label="Use Query Expansion",
                description = "Use query expansion to improve response context.",
                items = TransparentGPT_settings.query_expansion_options,
                initial_value="No query expansion"
            ),
        ]
    ).send()

@cl.on_settings_update
async def start(settings):
    TransparentGPT_settings.update_settings(settings)

@cl.on_message
async def handle_message(message: cl.Message):
    question = message.content
    expanded_query = ''
    if TransparentGPT_settings.query_expansion != 'No query expansion':
        if TransparentGPT_settings.query_expansion == 'Basic query expansion':
            t = 'Return a thorough but concise search term to answer this question: {question}'
            pt = PromptTemplate(input_variables=['question'], template=t)
            init_chain = pt | TransparentGPT_settings.llm
            expanded_query = init_chain.invoke({"question": question}).content
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
            expanded_query = ' '.join(init_chain.invoke({'question': message.content}))
        elif TransparentGPT_settings.query_expansion == "Hypothetical answer":
            hypothetical_answer = generate_hypothetical_answer(message.content)
            expanded_query = f'{message.content} {hypothetical_answer.content}'
    no_source_prompt=""
    if expanded_query == '' and not TransparentGPT_settings.display_sources:
        no_source_prompt = TransparentGPT_settings.prompt_mappings[TransparentGPT_settings.prompt_name+"_no_sources"]
        expanded_query = no_source_prompt.invoke({"question": question})
    elif expanded_query == '' and TransparentGPT_settings.display_sources:
        expanded_query = TransparentGPT_settings.prompt.invoke({"question":question})
    response = TransparentGPT_settings.llm.invoke(expanded_query)
    if no_source_prompt=="":
        similarity_values = []
        temp = response.content
        sources = []
        count = 0
        while "*" in temp:
            if count < 3:
                link_idx = temp.rfind("*")
                source = temp[link_idx+1:]
                similarity_values += [test_scrape_sim(source, response.content)]
                #score to source is reversed right now
                #collect sources more properly - bugging out right now if print title in comparison method
                #collect sources and then loop through
                response.content = response.content[:link_idx] + str(test_scrape_sim(source, response.content)) + response.content[link_idx:]
                temp = temp[:link_idx]
                count += 1
            else:
                break
    output_message = response.content + f"\n I am {highest_log_prob(response.response_metadata["logprobs"]['content'])}% confident in this response."
    await cl.Message(output_message).send()


if __name__ == '__main__':
    start()




# async def handle_message(message: cl.Message):
#     question = message.content
#     expanded_query = ''
#     if TransparentGPT_settings.query_expansion != 'No query expansion':
#         if TransparentGPT_settings.query_expansion == 'Basic query expansion':
#             t = 'Return a thorough but concise search term to answer this question: {question}'
#             pt = PromptTemplate(input_variables=['question'], template=t)
#             init_chain = pt | TransparentGPT_settings.llm
#             expanded_query = init_chain.invoke({"question": question}).content
#         elif TransparentGPT_settings.query_expansion == 'Multiquery expansion':
#             output_parser = LineListOutputParser()
#             pt = PromptTemplate(
#                 input_variables=['question'],
#                 template="""
#                 You are an AI language model assistant. Your task is to generate give different versions of the given user question to retrieve
#                 context for your response. By generating multiple perspectives on the user question, your goal is to help the user overcome
#                 some of hte limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. 
#                 Original question: {question},
#                 """
#             )
#             init_chain = pt | TransparentGPT_settings.llm | output_parser
#             expanded_query = ' '.join(init_chain.invoke({'question': message.content}))
#         elif TransparentGPT_settings.query_expansion == "Hypothetical answer":
#             hypothetical_answer = generate_hypothetical_answer(message.content)
#             expanded_query = f'{message.content} {hypothetical_answer.content}'
#     if not TransparentGPT_settings.display_sources:
#         no_source_prompt = TransparentGPT_settings.prompt_mappings[TransparentGPT_settings.prompt_name+"_no_sources"]
#         print("question: ", question)
#         if expanded_query == '':
#             response = no_source_prompt.invoke({"question": question})
#         else:
#             response = no_source_prompt.invoke({"question":expanded_query})
#         print("RESPONSE: ", response)
#     else:
#         if expanded_query == '':
#             response = TransparentGPT_settings.prompt.invoke({"question": question})    
#         else:
#             response = TransparentGPT_settings.llm.invoke(expanded_query)
#         print("RESPONSE: ", response)
#         similarity_values = []
#         temp = response.content
#         sources = []
#         count = 0
#         while "*" in temp:
#             if count < 3:
#                 link_idx = temp.rfind("*")
#                 source = temp[link_idx+1:]
#                 similarity_values += [test_scrape_sim(source, response.content)]
#                 response.content = response.content[:link_idx] + str(test_scrape_sim(source, response.content)) + response.content[link_idx:]
#                 temp = temp[:link_idx]
#                 count += 1
#             else:
#                 break


    # if expanded_query == '':
    #     expanded_query = TransparentGPT_settings.prompt.invoke({"question": question})
    # if not TransparentGPT_settings.display_sources:
    #     print("EXAPNDED QUERY: ", expanded_query)
    #     no_source_prompt = TransparentGPT_settings.prompt_mappings[TransparentGPT_settings.prompt_name+"_no_sources"]
        #prompt_value = no_source_prompt.invoke({"question": expanded_query})
        # response = no_source_prompt.invoke({"question": expanded_query})
        # print("NO SOURCE PROMPT: ", no_source_prompt.template)
        # #response = TransparentGPT_settings.llm.invoke(prompt_value)
        # response = TransparentGPT_settings.llm.invoke(no_source_prompt.template)
    # else:
    #     response = TransparentGPT_settings.llm.invoke(expanded_query)
    # similarity_values = []
    # temp = response.content
    # sources = []
    # count = 0
    # while "*" in temp:
    #     if count < 3:
    #         link_idx = temp.rfind("*")
    #         source = temp[link_idx+1:]
    #         similarity_values += [test_scrape_sim(source, response.content)]
    #         response.content = response.content[:link_idx] + str(test_scrape_sim(source, response.content)) + response.content[link_idx:]
    #         temp = temp[:link_idx]
    #         count += 1
    #     else:
    #         break
#     output_message = response.content + f"\n I am {highest_log_prob(response.response_metadata["logprobs"]['content'])}% confident in this response."
#     await cl.Message(output_message).send()

# if __name__ == '__main__':
#     start()
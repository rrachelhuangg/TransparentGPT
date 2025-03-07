import os
import chainlit as cl
from langchain.memory.buffer import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains import LLMChain
from prompts import default_prompt_template, doctor_prompt_template, default_prompt_template_no_sources, doctor_prompt_template_no_sources
from dotenv import load_dotenv
from chainlit.input_widget import Select, Switch, Slider
from langchain_core.prompts import ChatPromptTemplate
from math import exp
import numpy as np

#setting environment variables (non-Nebius API access keys)
load_dotenv()

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

TransparentGPT_settings = TransparentGPTSettings()

def highest_log_prob(vals):
    #returns the average log prob (confidence) of each token relative to the whole response token sequence
    logprobs = []
    for token in vals:
        logprobs += [token['logprob']]
    average_log_prob = sum(logprobs)/len(logprobs)
    return np.round(np.exp(average_log_prob)*100,2)

@cl.on_chat_start
async def start():
    await cl.Message("Hello! I am Parker, your social media analysis chatbot.🦋🕸️").send()
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
        ]
    ).send()

@cl.on_settings_update
async def start(settings):
    TransparentGPT_settings.update_settings(settings)

@cl.on_message
async def query_llm(message: cl.Message):
    if not TransparentGPT_settings.display_sources:
        no_source_prompt = TransparentGPT_settings.prompt_mappings[TransparentGPT_settings.prompt_name+"_no_sources"]
        prompt_value = no_source_prompt.invoke({"question": message.content})
        response = TransparentGPT_settings.llm.invoke(prompt_value)
        output_message = response.content + f"\n I am {highest_log_prob(response.response_metadata["logprobs"]['content'])}% confident in this response."
        await cl.Message(output_message).send()
    else:
        prompt_value = TransparentGPT_settings.prompt.invoke({"question": message.content})
        response = TransparentGPT_settings.llm.invoke(prompt_value)
        output_message = response.content + f"\n I am {highest_log_prob(response.response_metadata["logprobs"]['content'])}% confident in this response."
        await cl.Message(output_message).send()

if __name__ == '__main__':
    start()
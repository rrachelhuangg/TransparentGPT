"""
TransparentGPT Settings class and helper class to assist with multiquery expansion.
"""
import os
from langchain_core.output_parsers import BaseOutputParser
from typing import List
from prompts import default_prompt_template, doctor_prompt_template, default_prompt_template_no_sources, doctor_prompt_template_no_sources, default_quirky_genz_prompt, default_quirky_genz_prompt_no_sources, default_food_critic_prompt, default_food_critic_prompt_no_sources, default_media_critic_prompt, default_media_critic_prompt_no_sources
from methods import load_config
from langchain_openai import ChatOpenAI


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser that splits a LLM result into a list of queries."""
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split('\n')
        return list(filter(None, lines))

class TransparentGPTSettings:
    """
    Stores and handles TransparentGPT settings in persistent memory across files.
    """
    def __init__(self):
        self.model = "meta-llama/Llama-3.3-70B-Instruct"
        self.temperature = 0.7
        self.prompt = default_prompt_template
        self.prompt_mappings = {"default": default_prompt_template, "default_no_sources": default_prompt_template_no_sources, "doctor": doctor_prompt_template, "doctor_no_sources": default_prompt_template_no_sources, "genz": default_quirky_genz_prompt, "genz_no_sources": default_quirky_genz_prompt_no_sources, "food_critic": default_food_critic_prompt, "food_critic_no_sources": default_food_critic_prompt_no_sources, "media_critic":default_media_critic_prompt, "media_critic_no_sources": default_media_critic_prompt_no_sources }
        self.model_mappings = {"Meta Llama 3.1":"meta-llama/Meta-Llama-3.1-70B-Instruct", "Meta Llama 3.3":"meta-llama/Llama-3.3-70B-Instruct", "MistralAI":"mistralai/Mixtral-8x7B-Instruct-v0.1", "Dolphin Mixtral":"cognitivecomputations/dolphin-2.9.2-mixtral-8x22b", "Microsoft Mini":"microsoft/Phi-3-mini-4k-instruct"}
        self.prompt_name = "default"
        self.num_sources = load_config()["num_sources"]
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
        self.model = self.model_mappings[settings['Model']]
        self.temperature = settings['Temperature']
        self.prompt = self.prompt_mappings[settings['Prompt Template']]
        self.num_sources=settings['Number of Sources']
        self.llm = ChatOpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
            model = self.model,
            temperature = self.temperature
        ).bind(logprobs=True)
        self.display_sources = settings['Display Sources']
        self.prompt_name = settings['Prompt Template']
        self.query_expansion = settings['Query Expansion']

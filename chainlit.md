## Hi, I'm TransparentGPT! 🚀❤️‍🔥

I am a chatbot that is able to clarify my reasoning, explain my thought process, and cite the sources that I used for my response. 

I provide intermediate responses when I receive your query, expand upon that query, and am looking for sources to generate my response.

The sources that I use for my response are exposed by 🫐 **prompt engineering** 🫐, which means I am told to only use sources that I can provide links for. You can then click on these links to verify if the response I provided was correct or not. Along with each source, I output a percentage that shows how helpful that source was to my response.*

I also provide a percentage of how confident I am in each answer.**

In the Settings UI that can be found by clicking the gear icon in the message box, I provide a suite of customizable features!

Picking a 🪼 **large language model** 🪼 (LLM):
Large language models are AI systems that allow me to understand and output human-like natural language. You can pick from a handful of Meta, MistralAI, and Microsoft's open-source LLM's!

Display sources:
You can choose if you want me to display what 💎 **sources** 💎 I used for my response. I show my sources by default.

Number of sources:
You can choose the number of sources you want me to use for my response. 

Prompt template:
You can choose which 🦋 **prompt template** 🦋 I use. A prompt template essentially tells me how to act and speak when I respond to you.You can experiment with a doctor, genz, food-critic, and media-critic prompt template!

Query expansion:
You can choose if you want to use 🥶 **query expansion** 🥶. If so, you can also choose what kind of query expansion I use! Query expansion allows me to improve my response by "secretly" expanding your query with related terms behind-the-scenes. This allows me to find more relevant sources. 

Temperature:
You can choose how 🏙 **consistent** 🏙 my responses will be. 

#### I am built with Langchain, Chainlit, Nebius Studio, Open-source Large Language Models, a bit of web scraping, and some vector similarity analysis on text embeddings.

- **My Github:** [TransparentGPT](https://github.com/rrachelhuangg/TransparentGPT) 🔗

\* Calculating the relevance of each score: The bot is only allowed to use Wikipedia sources, for which direct links can be provided and the text content of the page is more easily scraped. I scrape the text content of the source using the Revisions API. Then, I compare the scraped source text and my response text via a cosine vector similarity analysis. Please note that the relevance scores provided with each source will seem pretty low because the scraped text of the sources is likely longer than my response and may contain HTML or other related 
formatting text, though I try to minimize non-natural language text in the scraped content. 

** Calculating how confident I am in my answer: I calculate the negative log-likelihood of each token in my response, relative to the position of other tokens. The average of these values can represent how confident I am in my response, and this form of confidence calculating is formally called "perplexity."


# TransparentGPT
* Chatbot that is able to <b>clarify its reasoning</b> 🧠, <b>explain its thought process</b> 🙊, and <b>cite the sources</b> 📚 that it used for its responses.
* Provides intermediate responses when user query is received, query is being expanded, and when sources are being retrieved to generate responses
* Sources that TransparentGPT uses are exposed by <b>prompt engineering</b>. Allows user to click on these source links to verify if responses are correct*
  * A <b>relevance score</b> is calculated and provided provided for each source to show the user how useful that source was in generating the TranparentGPT's response**
* A <b>confidence score</b> is also calculated and provided for each response***
* ReadME file in top right corner of TransparentGPT's UI provides a detailed explanation of the features and concepts descriped in this Github ReadME
* Settings panel provides a suite of <b>customizable features</b>

  <img width="400" alt="TransparentGPT_settings_panel" src="https://github.com/user-attachments/assets/46be3f5c-d795-438d-b110-7ed49b3d3b9b" />
  
  * Large language model (LLM): choose betweeen a handful of Meta, MistralAI, and Microsoft's open-source LLM's!
  * Source display: toggle switch allows user to choose if they want the sources used displayed with each response
  * Number of sources: slider allows user to choose how many sources they want to be used for each response
  * Prompt template: user can choose a prompt template for their bot. This determines what kind of TransparentGPT bot they interact with
  * Query expansion: user can choose between no query expansion, basic query expansion, multiquery query expansion, and hypothetical answer query expansion****
  * Temperature: slider allows user to choose how consistent they want their responses to be


I am built with Langchain, Chainlit, Nebius Studio, Open-source Large Language Models, a bit of web scraping, and some vector similarity analysis on text embeddings.
My Github: TransparentGPT 🔗
* Calculating the relevance of each score: The bot is only allowed to use Wikipedia sources, for which direct links can be provided and the text content of the page is more easily scraped. I scrape the text content of the source using the Revisions API. Then, I compare the scraped source text and my response text via a cosine vector similarity analysis. Please note that the relevance scores provided with each source will seem pretty low because the scraped text of the sources is likely longer than my response and may contain HTML or other related
formatting text, though I try to minimize non-natural language text in the scraped content.
** Calculating how confident I am in my answer: I calculate the negative log-likelihood of each token in my response, relative to the position of other tokens. The average of these values can represent how confident I am in my response, and this form of confidence calculating is formally called "perplexity."
* provide custom statistics about followers, activity, etc.

## Demo Video:

## Testing:
* `git clone https://github.com/rrachelhuangg/TransparentGPT.git`
* `cd social-media-chatbot`
* `pip install requirements.txt`
* `chainlit run chatbot.py -w --port 8000`
* Navigate to [localhost:8000](localhost:8000)

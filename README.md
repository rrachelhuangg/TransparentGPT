# TransparentGPT
* Chatbot that is able to <b>clarify its reasoning</b> 🧠, <b>explain its thought process</b> 🙊, and <b>cite the sources</b> 📚 that it used for its responses.
* Provides intermediate responses when user query is received, query is being expanded, and when sources are being retrieved to generate responses
* Sources that TransparentGPT uses are exposed by <b>prompt engineering</b>. Allows user to click on these source links to verify if responses are correct*
  * A <b>relevance score</b> is calculated and provided provided for each source to show the user how useful that source was in generating the TranparentGPT's response**
* A <b>confidence score</b> is also calculated and provided for each response***
* ReadME file in top right corner of TransparentGPT's UI provides a detailed explanation of the features and concepts descriped in this Github ReadME
* Settings panel provides a suite of <b>customizable features</b>

  <img width="400" alt="TransparentGPT_settings_panel" src="https://github.com/user-attachments/assets/46be3f5c-d795-438d-b110-7ed49b3d3b9b" />
  
  * <b>Large language model (LLM)</b>: choose betweeen a handful of Meta, MistralAI, and Microsoft's open-source LLM's!
  * <b>Source display</b>: toggle switch allows user to choose if they want the sources used displayed with each response
  * <b>Number of sources</b>: slider allows user to choose how many sources they want to be used for each response
  * <b>Prompt template</b>: user can choose a prompt template for their bot. This determines what kind of TransparentGPT bot they interact with
  * <b>Query expansion</b>: user can choose between no query expansion, basic query expansion, multiquery query expansion, and hypothetical answer query expansion****
  * <b>Temperature</b>: slider allows user to choose how consistent they want their responses to be

* Example of querying the Meta Llama 31 model with the GenZ prompt template bot using basic query expansion:

<img width="410" alt="Screenshot 2025-03-10 at 4 02 17 PM" src="https://github.com/user-attachments/assets/a668081d-49b7-4b50-9d16-ed89a24e327d"/> <img width="410" alt="Screenshot 2025-03-10 at 4 02 22 PM" src="https://github.com/user-attachments/assets/8e2b2c3a-9c59-4105-bfb2-21939cad9dba" />

![Static Badge](https://img.shields.io/badge/Langchain-green) ![Static Badge](https://img.shields.io/badge/Chainlit-red) ![Static Badge](https://img.shields.io/badge/NebiusStudio-black) ![Static Badge](https://img.shields.io/badge/MetaLlama-blue) ![Static Badge](https://img.shields.io/badge/MistralAI-purple) ![Static Badge](https://img.shields.io/badge/DolphinMixtral-blue) ![Static Badge](https://img.shields.io/badge/MicrsoftMini-black)

\* TransparentGPT is told to only use Wikipedia sources that it can provide links for. This is to demonstrate how sources can be accessed and analyzed, and can be expanded to include sources other than Wikipedia.

\** The relevance of each score is calculcated by scraping the text content of a source using the Revisions API. The scraped source text and TransparentGPT's response text are compared via a consine vector similarity analysis. Please note that the relevance scores provided with each source will seem pretty low because the scraped text of the sources is likely longer than the output response and may contain HTML or other related
formatting text, though non-natural language text is minimzed in the scraped content.

\*** The confidence of my answer is calculated by averaging the negative log-likelihood of each token in a response, relative to the position of other tokens. This represents the confidence of my response and is called the "perplexity."

\**** Query expansion is a method of improving chatbot responses by adding related terms and phrases to the user's query. This gives the chatbot more information to compare against when looking for helpful sources to use for its response. Multiquery query expansion is a method of query expansion where the large language model will generate n-related queries to use in addition to the user's query. TransparentGPT will generate 3 extra queries for multiquery expansion. For, hypothetical answer query expansion, the large language model will answer the user's query with no external context. This often means that it will make up information based on the user query. This is not used for the chatbot's final response, but rather improves its response, because the hypothetical answer will contain more relevant terms and phrases to help the chatbook look for helpful sources.

## Demo Video:

## Testing:
* `git clone https://github.com/rrachelhuangg/TransparentGPT.git`
* `cd social-media-chatbot`
* `pip install requirements.txt`
* Generate API keys:
  * Generate a Nebius Studio API key [here](https://studio.nebius.com/settings/api-keys)
  * Can also generate a LiteralAI API key [here](https://cloud.getliteral.ai/projects/rebuild_hackathon-yASBMe2aWvjB/settings?apiKeys-filter=%5B%5D) to enable immediate thumbs up/down feedback of each TransparentGPT response
* `chainlit run chatbot.py -w --port 8000`
* Navigate to [localhost:8000](http://localhost:8000/)

# Instructions

I would like for you to conduct some background research on the different AI companies that I am researching. In particular, focus on these companies: 

OpenAI
Anthropic
Google DeepMind
xAI
Meta AI
Cohere (Note - headquarters is Toronto, Canada - the California office is an affiliate)
Mistral
Baidu
DeepSeek
Alibaba Qwen

I want to know where the offices of these companies are. In particular, I am interested in knowing where all of the: 
- headquarters
- affiliate

I want to have this at the level of City, Country

To do this, I want you to do online research using the Serper API. You should do a serper websearch for each company. You can use proceed through one link at a time. When you find the answer, try to substantiate it in a separate source. If you cannot substantiate it after looking into 3 additional sources, then give up. 

We will be doing long-term, ongoing research on these companies. Therefore, it will be useful to save the search results so that we can query them again, should we need to. We should process the search results like we did with the Prosopography tool; we can adopt a similar provenance tracking system. The results should be saved in the database, perhaps as another schema, along the lines of a derivative, source material schema, with an affiliation to each company. In the future, I'd like to just be able to search i.e. sources on deepseek for information, or even add more sources to that as time goes on. 

We should chunk and embed sources - use Cohere for embeddings. 

For the RAG prompts, use Cohere Command-A with cohere Reranking. There is a cohere api key in the .env file. Make sure that you check for the api key in the .env file - do not look for it as a system variable. 

For the prompt pipeline - first look for the locations. When you have found some, then fact-check them. It's a two prompt system. I should be able to know which locations you could substantiate in two separate sources. 



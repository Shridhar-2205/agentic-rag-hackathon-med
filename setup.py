from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
import os
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from llama_index.readers.papers import PubmedReader
API_KEY = "5b6c5918-f3a7-4e57-b37d-a6951efc6748"       
os.environ["TAVILY_API_KEY"] = "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"




app = FastAPI()

llm = ChatOllama(
            model="llama3.2",
            temperature=0,
            verbose=True
        )

@tool
def qna(question : str) -> str:
    """Returns the answer to the question using the assistant"""
    pc = Pinecone(api_key=API_KEY)
    assistant = pc.assistant.Assistant(assistant_name='rag-a-thon')

    msg = Message(content=question)
    resp = assistant.chat(messages=[msg])
    return resp['message']['content']


tools = [qna]

prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "I will just make the qna call to get the answer to the question and improve the response, the output title should be Patient past visits information.", 
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt    
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@tool
def get_articles(disease_name):
    """This tool fetches articles from pubmed for a given disease name"""
    loader = PubmedReader()
    text_list = []
    page_title = []
    journal_title = []
    url_list = []
    map_data = []
    
    documents = loader.load_data(search_query=disease_name, max_results=4)
    for doc in documents:
      text_list.append(doc.text)
      page_title.append(doc.metadata['Title of this paper'])
      journal_title.append(doc.metadata['Journal it was published in:'])
      url_list.append(doc.metadata['URL'])
      map_data.append({"title": doc.metadata['Title of this paper'], "journal": doc.metadata['Journal it was published in:'], "url": doc.metadata['URL']})
      
    
    return map_data

tools_agent2 = [get_articles]
prompt_agent2 = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "just use the diseases and health condition names and give these three details with titles: URL, TITLE and JOURNAL explicitly of the articles based on the tool function response, format it in readable format and title of the response should be 'Suitable articles to refer for the patient', avoid notes and other unnecessary information",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
# pubmed articles agent
llm_with_tools_agent2 = llm.bind_tools(tools_agent2)

agent2 = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt_agent2    
    | llm_with_tools_agent2
    | OpenAIToolsAgentOutputParser()
)

# Create an agent executor by passing in the agent and tools
agent_executor2 = AgentExecutor(agent=agent2, tools=tools_agent2, verbose=True)

# Agent3
tools_agent3 = [TavilySearchResults(max_results=2)]
prompt_agent3 = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Use tavily search to get the information about the disease and health condition based on the input string, the output title should be 'Useful search results for the patient'",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

# tavily search agent
llm_with_tools_agent3 = llm.bind_tools(tools_agent3)

agent3 = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt_agent3  
    | llm_with_tools_agent3
    | OpenAIToolsAgentOutputParser()
)

# Create an agent executor by passing in the agent and tools
agent_executor3 = AgentExecutor(agent=agent3, tools=tools_agent3, verbose=True)
mock_response = "**Patient Past Visits Information**  Below is a summary of John Smith's past medical visits:  1.  **Routine Check-Up on 2024-06-15**     - Blood Pressure: 150/85 mmHg     - Heart Rate: 68 bpm     - Respiratory Rate: 18 breaths/min     - Temperature: 98.1°F     - Weight: 195 lbs     - BMI: 29.0     - HbA1c: 7.8%     - Creatinine: 1.4 mg/dL     - Sodium: 137 mEq/L     - Potassium: 4.1 mEq/L     - Cholesterol Total: 205 mg/dL     - LDL: 130 mg/dL"

@app.get("/agent1")
async def agent1():
    # result = agent_executor.invoke({"input": "Give patient details"})
    # return {"items": result['output']}
    return {"items": mock_response}

@app.get("/agent2")
async def agent2():
    result2 = agent_executor2.invoke({"input": "Hypertension"})
    agent2_output = result2['output']
    return {"items": agent2_output}

@app.get("/agent3")
async def agent3():
    result3 = agent_executor3.invoke({"input": "Diabetes"})
    agent3_output = result3['output']
    return {"message": agent3_output}

@app.get("/multiagent")
async def multiagent():
    result1 = agent_executor.invoke({"input": "Give patient details"})
    agent1_output = result1['output']
    result2 = agent_executor2.invoke({"input": {agent1_output}})
    result3 = agent_executor3.invoke({"input": {agent1_output}})
    agent3_output = result3['output']
    return {"message": agent3_output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from seleniumsearch import SeleniumSearchTool

llm = ChatOpenAI(temperature=0)

# Load the tool configs that are needed.
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
# Load in the Selenium Search Tool first
tools = [SeleniumSearchTool()]

from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    question: str = Field()

tools.append(
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
        # coroutine= ... <- you can specify an async method if desired as well
    ),
)

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools, 
    llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")

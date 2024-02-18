import os
import pandas as pd
from pdf import canada_engine
from dotenv import load_dotenv
from note_engine import note_engine
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from prompts import new_prompt, instruction_str, context
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata


# Load .env file
load_dotenv()

# Load population data
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

# Population Query Engine
population_query_engine = PandasQueryEngine(population_df, verbose=True, instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, 
                    metadata=ToolMetadata(
                        name="population_data", 
                        description="This tool allows you to query the population data.")),
    QueryEngineTool(query_engine=canada_engine,
                    metadata=ToolMetadata(
                        name = "canada_data",
                        description="This tool allows you to query the Canada data.")),
]

llm = OpenAI(model="gpt-3.5-turbo")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit):")) != "q":
    result = agent.query(prompt)
    print(result)
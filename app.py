# Importing Required Libraries

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
from langchain.schema import AgentFinish
from langchain.schema.output import LLMResult

# SQL database tooling
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# Loads API keys from a .env file
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")



## Load Data and Save to SQLite Database
# Load the dataset
df = pd.read_csv("salaries.csv")
df.head()

# Save the dataframe to an SQLite database
connection = sqlite3.connect("salaries.db")
df.to_sql(name="salaries", con=connection, if_exists="replace", index=False)


# Setup the logger: This callback will log the prompts and responses from Llama 3 to a file
# Logging AI Model Interactions

# 1. Event class formats the log (event type, timestamp, text)
# 2. LLMCallbackHandler captures when the LLM:
#    - Starts (logs the prompt)
#    - Ends (logs the response)
# 3. Logs are stored in prompts.jsonl

@dataclass
class Event:
    event: str
    timestamp: str
    text: str


def _current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


class LLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        assert len(prompts) == 1
        event = Event(event="llm_start", timestamp=_current_time(), text=prompts[0])
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        generation = response.generations[-1][-1].message.content
        event = Event(event="llm_end", timestamp=_current_time(), text=generation)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")



# Setup the LLM with the logging callback and it also logs each prompt-response via the callback handler
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    callbacks=[LLMCallbackHandler(Path("prompts.jsonl"))]
)


# Prepare the configuration
# This defines a Python dictionary named config that stores metadata about the database connection
config = {
    "db_type": "sqlite",
    "db_path": "sqlite:///salaries.db"
}
         
# Establish a database connection
db = SQLDatabase.from_uri("sqlite:///salaries.db")


## Create tools =>
# The tools will be based on the langchain_community SQL database tools. 
# The tools will be wrapped using the @tool decorator to make them available to our CrewAI agents.

# Tool 1: List all the tables in the database
@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")

list_tables.run()


# Tool 2 : Return the schema and sample rows for a given list of tables
@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

print(tables_schema.run("salaries"))


# Tool 3 : checks the SQL query before executing it
@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

check_sql.run("SELECT * WHERE salary > 10000 LIMIT 5 table = salaries")


# Tool 4: Executes a given SQL query
@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

execute_sql.run("SELECT * FROM salaries WHERE salary > 10000 LIMIT 5")




## Create Agents =>

# Agent 1 : Database Developer Agent will construct and execute SQL queries
sql_dev = Agent(
    role="Senior Database Developer",
    goal="Construct and execute SQL queries based on a request",
    backstory=dedent(
        """
        You are an experienced database engineer who is master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries.
        Use the `list_tables` to find available tables.
        Use the `tables_schema` to understand the metadata for the tables.
        Use the `execute_sql` to check your queries for correctness.
        Use the `check_sql` to execute queries against the database.
    """
    ),
    llm=llm,
    tools=[list_tables, tables_schema, check_sql, execute_sql],
    allow_delegation=False,
)


# Agent 2 : Data Analyst Agent will analyze the database data response and write a detailed response
data_analyst = Agent(
    role="Senior Data Analyst",
    goal="You receive data from the database developer and analyze it",
    backstory=dedent(
        """
        You have deep experience with analyzing datasets using Python.
        Your work is always based on the provided data and is clear,
        easy-to-understand and to the point. You have attention
        to detail and always produce very detailed work (as long as you need).
    """
    ),
    llm=llm,
    allow_delegation=False,
)


# Agent 3 : Report Editor Agent will write an executive summary based on the analysis
report_writer = Agent(
    role="Senior Report Editor",
    goal="Write an executive summary type of report based on the work of the analyst",
    backstory=dedent(
        """
        Your writing still is well known for clear and effective communication.
        You always summarize long texts into bullet points that contain the most
        important details.
        """
    ),
    llm=llm,
    allow_delegation=False,
)



## Create Tasks =>

# Task 1 : Extract the data required for the user query
extract_data = Task(
    description="Extract data that is required for the query {query}.",
    expected_output="Database result for the query",
    agent=sql_dev,
)


# Task 2 : Analyze the data from the database and write an analysis
analyze_data = Task(
    description="Analyze the data from the database and write an analysis for {query}.",
    expected_output="Detailed analysis text",
    agent=data_analyst,
    context=[extract_data],
)


# Task 3 : Write an executive summary of the report from the analysis
write_report = Task(
    description=dedent(
        """
        Write an executive summary of the report from the analysis. The report
        must be less than 100 words.
    """
    ),
    expected_output="Markdown report",
    agent=report_writer,
    context=[analyze_data],
)



# Setup The Crew: Initialize the Crew with agents and tasks
crew = Crew(
    agents=[sql_dev, data_analyst, report_writer],
    tasks=[extract_data, analyze_data, write_report],
    process=Process.sequential,
    verbose=True,
    memory=False,
    output_log_file="crew.log",
)


# Kickoff the Crew
inputs = {
    "query": "What is the name of the job title of lowest and highest salary in USD?"
}

result = crew.kickoff(inputs=inputs)
print(result)

# streamlit_app.py

import streamlit as st
import pandas as pd
import sqlite3
import os
from pathlib import Path
import json
from textwrap import dedent
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool, QuerySQLDataBaseTool
)
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_groq import ChatGroq

# Load API Key
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Session State Setup
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "llm" not in st.session_state:
    st.session_state.llm = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Callback and Logging
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

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any):
        event = Event(event="llm_start", timestamp=_current_time(), text=prompts[0])
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")

    def on_llm_end(self, response, **kwargs: Any):
        generation = response.generations[-1][-1].message.content
        event = Event(event="llm_end", timestamp=_current_time(), text=generation)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")

# UI Title
st.title("📊 CSV to SQL Agent App")

# Load LLM Section
if not st.session_state.model_loaded:
    if st.button("⚙️ Load LLM Model"):
        with st.spinner("Loading model..."):
            st.session_state.llm = ChatGroq(
                temperature=0,
                model_name="llama3-70b-8192",
                callbacks=[LLMCallbackHandler(Path("prompts.jsonl"))]
            )
            st.session_state.model_loaded = True
        st.success("✅ Model loaded successfully!")

# Upload CSV Section
if st.session_state.model_loaded:
    st.subheader("📁 Step 1: Upload CSV Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Preview of Uploaded Dataset", df.head())

        # Save to SQLite
        conn = sqlite3.connect("data.db")
        df.to_sql(name="user_table", con=conn, if_exists="replace", index=False)

        # Save DB object to session
        st.session_state.db = SQLDatabase.from_uri("sqlite:///data.db")
        st.session_state.data_loaded = True
        st.success("✅ Dataset uploaded and stored in SQLite")

# Show query input section after both model and dataset are loaded
if st.session_state.model_loaded and st.session_state.data_loaded:
    st.subheader("💬 Step 2: Ask a Question")

    query_input = st.text_input("Type your query below:")

    # Define Tools
    db = st.session_state.db

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


    # Tool 3: Executes a given SQL query
    @tool("execute_sql")
    def execute_sql(sql_query: str) -> str:
        """Execute a SQL query against the database. Returns the result"""
        return QuerySQLDataBaseTool(db=db).invoke(sql_query)

    execute_sql.run("SELECT * FROM salaries WHERE salary > 10000 LIMIT 5")


    # Tool 4 : checks the SQL query before executing it
    @tool("check_sql")
    def check_sql(sql_query: str) -> str:
        """
        Use this tool to double check if your query is correct before executing it. Always use this
        tool before executing a query with `execute_sql`.
        """
        return QuerySQLCheckerTool(db=db, llm=st.session_state.llm).invoke({"query": sql_query})

    check_sql.run("SELECT * WHERE salary > 10000 LIMIT 5 table = salaries")



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
        llm=st.session_state.llm,
        tools=[list_tables, tables_schema, execute_sql, check_sql],
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
        llm=st.session_state.llm,
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
        llm=st.session_state.llm,
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

    if st.button("🔍 Ask a Question"):
        if query_input.strip():
            with st.spinner("Running the query..."):
                result = crew.kickoff(inputs={"query": query_input})
            st.subheader("🧠 Executive Summary")
            st.markdown(result)
        else:
            st.warning("⚠️ Please enter a query.")

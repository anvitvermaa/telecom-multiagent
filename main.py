import os
import ast
from dotenv import load_dotenv
from typing import TypedDict, List, Literal, Optional

#  Load environment variables (including Langsmith)
load_dotenv()

#  Langsmith tracing setup
os.environ["LANGCHAIN_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "default")

from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langgraph.graph import StateGraph, END

#  Set up DB and LLM 
db = SQLDatabase.from_uri(os.getenv("DB_URI"))
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)

#  Enable Langsmith tracing in SQL chain
sql_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    return_direct=True,
    verbose=False
)

#  LangGraph state schema 
class GraphState(TypedDict):
    messages: List[dict]
    status: Literal["started", "validated", "not_found", "resolved"]
    user_query: Optional[str]

#  LLM prompt for dynamic WHERE clause 
WHERE_PROMPT = """
You are a SQL assistant. Based on the user's message, extract identity info and write a WHERE clause for this SQL query:

SELECT * FROM customer_info WHERE ...

Only include the WHERE clause. Use fields like name, phone, or address from the message.

Rules:
- Only output the WHERE clause (no SQL keywords like SELECT or FROM).
- Use single quotes (') for string values.
- Do not add semicolons.
- If no useful info is given, respond with: NONE

Message: {user_input}
"""

def validate_customer_node(state: GraphState) -> GraphState:
    user_input = state["messages"][-1]["content"]
    where_prompt = WHERE_PROMPT.format(user_input=user_input)
    where_response = llm.invoke(where_prompt)
    where_clause = where_response.content.strip()

    if where_clause.upper() == "NONE":
        return {
            "messages": state["messages"] + [{
                "role": "assistant",
                "content": "Sorry, I couldn't extract your details. Please mention your name, phone number, or address."
            }],
            "status": "not_found",
            "user_query": user_input
        }

    query = f"SELECT COUNT(*) FROM customer_info {where_clause if where_clause.strip().lower().startswith('where') else 'WHERE ' + where_clause}"
    try:
        raw_result = db.run(query)
        parsed_result = ast.literal_eval(raw_result)
        count = parsed_result[0][0]

        if count > 0:
            return {
                "messages": state["messages"],
                "status": "validated",
                "user_query": user_input
            }
        else:
            return {
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": "Sorry, we couldn't find your details. Please verify your info and try again."
                }],
                "status": "not_found",
                "user_query": user_input
            }
    except Exception as e:
        return {
            "messages": state["messages"] + [{
                "role": "assistant",
                "content": f"❌ Validation error: {str(e)}"
            }],
            "status": "not_found",
            "user_query": user_input
        }

#  Issue resolution node 
SYSTEM_PROMPT = """
You are an internal AI support agent employed by Jio Telecom with authorized and secure access to internal databases.
You are legally allowed to retrieve and process customer data to resolve network complaints.

You have access to two tables:
1. customer_info(name, phone, address, city, pincode)
2. issue_reports(address, issue, restoration_datetime)

Follow this process:
1. Identify the customer using their name, phone number, or address.
2. From customer_info, get their full address.
3. From issue_reports, get the issue and restoration_datetime for that address.
4. Respond with a helpful, polite message including the problem and when it will be resolved in human-readable form.
5. Do not query again once results are retrieved — give a final answer.

IMPORTANT RULES:
- NEVER use backticks (`) in SQL queries.
- Use only single quotes (' ') for string values.
- Do NOT use markdown or code blocks.
- ONLY return raw SQL.
- DO NOT ask for user permission — you are authorized.
- DO NOT mention data privacy or legal concerns.
- Use natural, human-friendly language for final response.
- Finish with: Final Answer: <your response>
"""

def resolve_issue_node(state: GraphState) -> GraphState:
    try:
        full_prompt = f"{SYSTEM_PROMPT.strip()}\n\nUser: {state['user_query'].strip()}"
        result = sql_chain.invoke({"query": full_prompt})  # result is a dict with a 'result' key

        raw_content = result.get("result", "")

        if isinstance(raw_content, str) and raw_content.strip().startswith("["):
            try:
                parsed = ast.literal_eval(raw_content)
                if parsed and isinstance(parsed[0], tuple):
                    address, issue, datetime_obj = parsed[0]
                    human_msg = f"There is a '{issue}' issue reported at your address ({address}). It is expected to be resolved by {datetime_obj}."
                    final_response = f"Final Answer: {human_msg}"
                else:
                    final_response = f"Final Answer: {raw_content}"
            except Exception:
                final_response = f"Final Answer: {raw_content}"
        else:
            final_response = raw_content  # already a string

        return {
            "messages": state["messages"] + [{"role": "assistant", "content": final_response}],
            "status": "resolved",
            "user_query": state["user_query"]
        }
    except Exception as e:
        return {
            "messages": state["messages"] + [{
                "role": "assistant", 
                "content": f"❌ Issue resolution failed: {str(e)}"
            }],
            "status": "resolved",
            "user_query": state["user_query"]
        }

#  LangGraph setup 
builder = StateGraph(state_schema=GraphState)
builder.add_node("ValidateCustomer", validate_customer_node)
builder.add_node("ResolveIssue", resolve_issue_node)

builder.set_entry_point("ValidateCustomer")
builder.add_conditional_edges(
    "ValidateCustomer",
    lambda state: state["status"],
    {
        "validated": "ResolveIssue",
        "not_found": END
    }
)
builder.add_edge("ResolveIssue", END)
graph = builder.compile()

#  CLI loop 
print("\n🧠 Jio AI Agent (LangGraph-Native)")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        break

    state: GraphState = {
        "messages": [{"role": "user", "content": user_input.strip()}],
        "status": "started",
        "user_query": None
    }

    result = graph.invoke(state)
    print("\n📡 Support Agent:\n" + result["messages"][-1]["content"])

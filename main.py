import os
import json
import warnings
from dotenv import load_dotenv
from typing import Annotated, Literal

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Langsmith tracing setup
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "default")

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, MessagesState

# RAG Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 1. Initialize RAG (Vector Database)
print("Initializing ChromaDB and loading Jio Manuals...")
# Using OllamaEmbeddings instead of HuggingFace to save disk space
embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL", "llama3.1"))

if os.path.exists("jio_manual.txt"):
    loader = TextLoader("jio_manual.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
else:
    print("Warning: jio_manual.txt not found.")
    retriever = None

# Set up LLM 
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0)

# 2. Define Structured Output Schema
class CustomerInfoExtraction(BaseModel):
    """Extract customer details from their query."""
    is_info_present: bool = Field(description="True if the user provided a name, phone, or address.")
    name: str | None = Field(default=None, description="The customer's name if provided")
    phone: str | None = Field(default=None, description="The customer's phone if provided")
    address: str | None = Field(default=None, description="The customer's address if provided")

# LLM with structured output enabled
extractor_llm = llm.with_structured_output(CustomerInfoExtraction)

def load_customers():
    try:
        with open("customers.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def validate_customer_node(state: MessagesState) -> dict:
    user_message = state["messages"][0].content
    
    try:
        # Extract structured details reliably
        extraction = extractor_llm.invoke(user_message)
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"❌ I'm sorry, I had trouble processing your request. Please try again.")]
        }

    if not extraction.is_info_present or (not extraction.name and not extraction.phone and not extraction.address):
        return {
            "messages": [AIMessage(content="Sorry, I couldn't extract your details. Please mention your name, phone number, or address.")]
        }
        
    # Search local JSON DB
    customers = load_customers()
    found_customer = None
    
    for c in customers:
        if extraction.name and extraction.name.lower() in c.get("name", "").lower():
            found_customer = c
            break
        if extraction.phone and extraction.phone in c.get("phone", ""):
            found_customer = c
            break
        if extraction.address and extraction.address.lower() in c.get("address", "").lower():
            found_customer = c
            break

    if found_customer:
        # Store customer record text in a context message so the next node can use it
        return {
            "messages": [AIMessage(content=f"SYSTEM: Validated Customer. Data: {json.dumps(found_customer)}")]
        }
    else:
        return {
            "messages": [AIMessage(content="Sorry, we couldn't find your details. Please verify your info and try again.")]
        }

def route_after_validation(state: MessagesState) -> Literal["ResolveIssue", "__end__"]:
    last_msg = state["messages"][-1].content
    if last_msg.startswith("SYSTEM: Validated Customer."):
        return "ResolveIssue"
    return "__end__"

SYSTEM_PROMPT = """
You are an internal AI support agent employed by Jio Telecom.

Customer Record:
{customer_data}

Troubleshooting Manual (RAG Context):
{manual_context}

User Query: {user_query}

Process:
1. Review the Customer Record. If there is an active network issue affecting their address (e.g. area outage), inform them about it and apologize for the inconvenience. Do NOT try to troubleshoot if it's an area outage.
2. If there are NO active issues in their record, use the Troubleshooting Manual to answer their technical query and guide them step-by-step.
3. If their technical issue is not mentioned in the manual, inform them that a technician will be scheduled to visit their address.

IMPORTANT RULES:
- Use natural, human-friendly language.
- DO NOT mention JSON or database structures to the user.
- Keep the response direct and helpful.
"""

def resolve_issue_node(state: MessagesState) -> dict:
    last_msg = state["messages"][-1].content
    customer_data = last_msg.replace("SYSTEM: Validated Customer. Data: ", "")
    user_query = state["messages"][0].content
    
    try:
        # Retrieve context from Vector DB (RAG)
        manual_context = "No manual available."
        if retriever:
            docs = retriever.invoke(user_query)
            manual_context = "\n\n".join([d.page_content for d in docs])
        
        full_prompt = SYSTEM_PROMPT.format(
            customer_data=customer_data,
            manual_context=manual_context,
            user_query=user_query
        )
        
        response = llm.invoke(full_prompt)
        return {
            "messages": [AIMessage(content=response.content)]
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"❌ Issue resolution failed: {str(e)}")]
        }


# LangGraph setup 
builder = StateGraph(MessagesState)
builder.add_node("ValidateCustomer", validate_customer_node)
builder.add_node("ResolveIssue", resolve_issue_node)

builder.set_entry_point("ValidateCustomer")
builder.add_conditional_edges("ValidateCustomer", route_after_validation)
builder.add_edge("ResolveIssue", END)

graph = builder.compile()

if __name__ == "__main__":
    print("\n🧠 Jio AI Agent (ChromaDB RAG + JSON Powered)")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break

            state = {"messages": [HumanMessage(content=user_input.strip())]}
            
            result = graph.invoke(state)
            
            print("\n📡 Support Agent:\n" + result["messages"][-1].content)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ System Error: {e}")

import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun  # <--- Check this line

# --- CONFIGURATION ---
# We use Llama 3 (70B) because it is free on Groq and very smart
llm = ChatGroq(temperature=0, groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama3.3-70b-versatile")
search = DuckDuckGoSearchRun()

# --- DEFINE THE FOLDER (STATE) ---
class AgentState(TypedDict):
    topic: str
    research_data: str
    article: str

# --- STEP 1: RESEARCHER ---
def researcher_node(state):
    print(f"🕵️ Researching: {state['topic']}")
    # Searches the web for the latest info
    try:
        results = search.run(f"{state['topic']} news latest")
    except:
        results = "Search failed, using general knowledge."
    return {"research_data": results}

# --- STEP 2: WRITER ---
def writer_node(state):
    print("✍️ Writing...")
    prompt = f"""
    You are an expert tech journalist.
    TOPIC: {state['topic']}
    RESEARCH: {state['research_data']}
    
    Task: Write a short, engaging markdown blog post.
    Include a Title and 3 Bullet points.
    """
    response = llm.invoke(prompt)
    return {"article": response.content}

# --- CONNECT THE STEPS ---
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)

workflow.set_entry_point("Researcher")
workflow.add_edge("Researcher", "Writer")
workflow.add_edge("Writer", END)

app = workflow.compile()

# --- RUN THE AGENT ---
if __name__ == "__main__":
    # You can change this topic to whatever you want
    topic = "Latest Artificial Intelligence breakthroughs 2025"
    print(f"🚀 Starting Agent on topic: {topic}")
    
    result = app.invoke({"topic": topic})
    
    # Save the output to a file so we can see it
    os.makedirs("content", exist_ok=True)
    with open("content/daily_post.md", "w") as f:
        f.write(result['article'])
        
    print("✅ Finished! content/daily_post.md created.")

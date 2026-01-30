import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Define the "State" (What the agents share)
class AgentState(TypedDict):
    topic: str
    research: str
    outline: List[str]
    content: str
    iteration: int

# 2. Initialize the model (Llama 3.3 70B is best for logic)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
search = DuckDuckGoSearchRun()

# --- AGENT NODES ---

def researcher_node(state: AgentState):
    print(f"🕵️ Researcher deep-diving into {state['topic']}...")
    queries = [f"{state['topic']} facts 2026", f"{state['topic']} expert analysis", f"{state['topic']} statistics"]
    results = [search.run(q) for q in queries]
    return {"research": "\n\n".join(results)}

def architect_node(state: AgentState):
    print("📐 Architect building SEO/AIO outline...")
    prompt = f"Using this research: {state['research']}, create a 5-section SEO outline for an article on {state['topic']}. Return ONLY the section titles as a list."
    response = llm.invoke(prompt)
    sections = [s.strip() for s in response.content.split('\n') if s]
    return {"outline": sections}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer drafting Section: {current_section}...")
    prompt = f"""Write a comprehensive, professional blog section for: {current_section}. 
    Use the research: {state['research']}. 
    Focus: SEO-friendly H2 tags, short paragraphs, and 2026 AIO (AI Optimization) standards.
    Length: 400 words for this section."""
    response = llm.invoke(prompt)
    return {"content": state['content'] + "\n\n" + response.content, "iteration": state['iteration'] + 1}

# --- THE GRAPH (THE TASK FORCE) ---

workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "architect")
workflow.add_edge("architect", "writer")

# Logic to loop through the outline sections
def should_continue(state: AgentState):
    if state['iteration'] < len(state['outline']):
        return "writer"
    return END

workflow.add_conditional_edges("writer", should_continue)

app = workflow.compile()

# Run it
if __name__ == "__main__":
    topic = "Latest AI breakthroughs in 2026"
    final_state = app.invoke({"topic": topic, "content": "", "iteration": 0})
    
    # Save the mega-article
    with open(f"content/article.md", "w") as f:
        f.write(final_state['content'])

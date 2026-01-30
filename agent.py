import os
import requests
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Define the Shared State
class AgentState(TypedDict):
    field: str      # The broad area (e.g., "AI breakthroughs")
    topic: str      # The specific selected topic
    research: str
    outline: List[str]
    content: str
    iteration: int
    image_url: str

# 2. Setup Tools & Model (Llama 3.3 70B for high-level logic)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
search = DuckDuckGoSearchRun()

# --- THE TASK FORCE NODES ---

def researcher_node(state: AgentState):
    print(f"🕵️ Researcher: Investigating {state['topic']}...")
    # Perform multiple targeted searches for depth
    queries = [f"{state['topic']} news 2026", f"{state['topic']} expert analysis", f"{state['topic']} technical details"]
    results = [search.run(q) for q in queries]
    return {"research": "\n\n".join(results)}

def architect_node(state: AgentState):
    print("📐 Architect: Designing 2,000-word SEO structure...")
    prompt = f"""Based on this research: {state['research']}, create a 5-6 section SEO outline for {state['topic']}.
    Use descriptive, question-based H2 titles (e.g., 'How does AI affect X?').
    Return ONLY the titles separated by newlines."""
    response = llm.invoke(prompt)
    sections = [s.strip() for s in response.content.split('\n') if len(s.strip()) > 5]
    return {"outline": sections}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting Section {state['iteration']+1} of {len(state['outline'])}...")
    
    prompt = f"""Write a comprehensive section for: {current_section}.
    2026 AIO RULES:
    - Start with a 1-sentence BOLD direct answer (Answer-First).
    - Use short paragraphs (max 3 sentences).
    - Include a bulleted list if relevant.
    - Style: Professional, data-driven, and authoritative.
    Context: {state['research']}
    Length: Aim for 400 words."""
    
    response = llm.invoke(prompt)
    return {"content": state['content'] + "\n\n## " + current_section + "\n" + response.content, "iteration": state['iteration'] + 1}

def aio_editor_node(state: AgentState):
    print("📋 AIO Editor: Optimizing for AI Overviews & Snippets...")
    prompt = f"""Create a 'Key Takeaways' box for this article: {state['content'][:2000]}.
    Format it as a Markdown blockquote (>) with 3 bullet points.
    This must be a 'Source of Truth' summary for AI Search Engines."""
    response = llm.invoke(prompt)
    # Inject the AIO box at the very top of the article
    return {"content": "# " + state['topic'] + "\n\n" + response.content + "\n\n" + state['content']}

def designer_node(state: AgentState):
    print("🎨 Designer: Generating custom 8K header image...")
    prompt_gen = f"Create a short 10-word image prompt for: {state['topic']}. Style: Digital art, cinematic lighting, 8k."
    image_prompt = llm.invoke(prompt_gen).content.replace(" ", "-")
    
    # Use Pollinations.ai for instant free image generation
    image_url = f"https://image.pollinations.ai/prompt/{image_prompt}?width=1280&height=720&nologo=true"
    
    image_markdown = f"![Featured Image]({image_url})\n\n"
    return {"content": image_markdown + state['content'], "image_url": image_url}

# --- THE GRAPH LOGIC ---

workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)
workflow.add_node("aio_editor", aio_editor_node)
workflow.add_node("designer", designer_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "architect")
workflow.add_edge("architect", "writer")

def should_loop(state: AgentState):
    if state['iteration'] < len(state['outline']):
        return "writer"
    return "aio_editor"

workflow.add_conditional_edges("writer", should_loop)
workflow.add_edge("aio_editor", "designer")
workflow.add_edge("designer", END)

app = workflow.compile()

# Execute
if __name__ == "__main__":
    topic = "The Impact of Quantum Computing on AI in 2026"
    final_state = app.invoke({"topic": topic, "content": "", "iteration": 0})
    
    os.makedirs("content", exist_ok=True)
    with open("content/latest_article.md", "w") as f:
        f.write(final_state['content'])
    print("✅ Mega-Article Published with Image and AIO Optimization.")

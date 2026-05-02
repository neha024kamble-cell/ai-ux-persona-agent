from crewai import Agent, Task, Crew, Process, LLM

# 1. Setup local AI (Ollama)
local_llm = LLM(model="ollama/llama3", base_url="http://localhost:11434")

# 2. Define the Agent
persona_architect = Agent(
    role='UX Research Specialist',
    goal='Generate high-quality user personas from raw data.',
    backstory='You are an expert at identifying patterns and building empathy.',
    llm=local_llm
)

# 3. List of Multiple Datasets (Add as many as you want)
datasets = [
    "Mark is 42, corporate traveler, hates pop-ups, wants dark mode.",
    "Sarah is 34, grocery shopper, frustrated by out-of-stock items.",
    "Leo is 25, fitness enthusiast, wants a workout app that syncs with his watch."
]

# 4. Loop to process each dataset
for i, note in enumerate(datasets):
    print(f"\n--- Processing Persona {i+1} ---\n")
    
    task = Task(
        description=f"Analyze these specific notes: {note}. Create a detailed user persona.",
        expected_output="A markdown document for the persona.",
        agent=persona_architect
    )

    crew = Crew(agents=[persona_architect], tasks=[task])
    result = crew.kickoff()
    
    # Optional: Save each persona to a separate text file
    with open(f"persona_{i+1}.md", "w") as f:
        f.write(str(result))
        
    print(f"Persona {i+1} saved as persona_{i+1}.md")
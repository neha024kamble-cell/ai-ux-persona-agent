from crewai import Agent, Task, Crew, Process, LLM

local_llm = LLM(model="ollama/llama3", base_url="http://localhost:11434")

# Specialist for extracting insights from messy text
analyst = Agent(
    role='Data Synthesis Specialist',
    goal='Identify key themes, pain points, and behaviors from raw, messy interview transcripts.',
    backstory='You are an expert at finding the signal in the noise. You ignore filler words and focus on core user needs.',
    llm=local_llm,
    verbose=True
)

# Specialist for building the final profile
architect = Agent(
    role='Persona Architect',
    goal='Use analyzed themes to create a realistic and empathetic user persona.',
    backstory='You turn raw data points into a human story that designers can use.',
    llm=local_llm,
    verbose=True
)

research_notes = """
So I'm talking to Mark, he's 42 and works in corporate sales, so he travels like twice a month for work but also tries to take a big family trip every summer. He told me he gets really overwhelmed by those giant comparison sites because there are just too many pop-ups and 'only 1 room left!' warnings that feel fake to him. He uses an Android phone, some Samsung model. His biggest frustration is actually the checkout process—he hates having to re-enter his corporate credit card and his personal one every single time, he wishes the app just knew which one was for which trip. He's a 'Value Seeker' but not necessarily looking for the cheapest, just the one with the best cancellation policy because his meetings move around a lot. He mentioned that he often books things at 11 PM when his kids are asleep, so he's usually tired and wants a 'dark mode' that actually works so he doesn't wake up his wife. He said, 'If I have to click more than four times to see the final price with taxes, I just close the tab and go somewhere else.'
"""

# Task 1: Clean and extract
extraction_task = Task(
    description=f"Extract specific goals, pain points, and tech habits from this messy note: {research_notes}",
    expected_output="A bulleted list of raw behavioral insights.",
    agent=analyst
)

# Task 2: Build from the clean list
persona_task = Task(
    description="Based on the extracted insights, create a high-fidelity User Persona document.",
    expected_output="A structured markdown document with sections for Biography, Goals, and Pain Points.",
    agent=architect,
    context=[extraction_task] # This links the two tasks together
)

ux_crew = Crew(
    agents=[analyst, architect],
    tasks=[extraction_task, persona_task],
    process=Process.sequential
)

print(ux_crew.kickoff())

# example_usage.py
"""
Example usage of the ResearchAgent.
"""
from research_agent.agent import ResearchAgent

def main():
    # Initialize the research agent
    agent = ResearchAgent(api_key="your_openai_api_key")
    
    # Define a research problem
    research_problem = "What are the environmental impacts of electric vehicles compared to traditional combustion engine vehicles?"
    
    # Conduct research
    result = agent.conduct_research(research_problem, citation_style="APA")
    
    # Save the report
    agent.save_report("ev_research_report.md")
    
    # Save the full research data including citations
    agent.save_research_data("ev_research_data.json")
    
    print(f"Research completed! Report saved to ev_research_report.md")

if __name__ == "__main__":
    main()

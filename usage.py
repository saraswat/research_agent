
# example_usage.py
"""
Example usage of the ResearchAgent.
"""
import os
import argparse
from research_agent.agent import ResearchAgent

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Research Agent")
    parser.add_argument("--backend", choices=["gemini", "openai"], default="gemini",
                      help="The LLM backend to use (gemini or openai)")
    parser.add_argument("--api-key", help="API key for the selected backend")
    parser.add_argument("--model", help="Model name to use (depends on backend)")
    parser.add_argument("--temperature", type=float, default=0.7, 
                      help="Temperature for response generation")
    parser.add_argument("--output-dir", default="output", 
                      help="Directory to save research outputs")
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key
    if not api_key:
        if args.backend == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
        else:  # openai
            api_key = os.environ.get("OPENAI_API_KEY")
            
    if not api_key:
        raise ValueError(f"No API key provided for {args.backend} backend. Either use --api-key or set the appropriate environment variable.")
        
    # Set default model if not provided
    model = args.model
    if not model:
        if args.backend == "gemini":
            model = "gemini-1.5-pro"
        else:  # openai
            model = "gpt-4o"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the research agent with the selected backend
    print(f"Initializing Research Agent with {args.backend.capitalize()} backend...")
    agent = ResearchAgent(
        api_key=api_key,
        backend=args.backend,
        model=model,
        temperature=args.temperature,
        top_p=0.95,  # Default top_p value
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints")
    )
    
    # Define a research problem
    research_problem = "What are the environmental impacts of electric vehicles compared to traditional combustion engine vehicles?"
    
    # Conduct research
    print(f"Starting research using {args.backend.capitalize()} {model}...")
    result = agent.conduct_research(research_problem, citation_style="APA")
    
    # Save the report
    report_path = os.path.join(args.output_dir, f"{args.backend}_research_report.md")
    agent.save_report(report_path)
    
    # Save the full research data including citations
    data_path = os.path.join(args.output_dir, f"{args.backend}_research_data.json")
    agent.save_research_data(data_path)
    
    print(f"Research completed! Report saved to {report_path}")
    print(f"Full research data saved to {data_path}")

if __name__ == "__main__":
    main()

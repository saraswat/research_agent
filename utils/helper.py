
# utils/helpers.py
"""
Helper utilities for the research agent.
"""
from typing import Dict, List, Any, Optional

def format_research_problem(problem: str) -> str:
    """
    Format a research problem for better processing by the agent.
    
    Args:
        problem: The original research problem
        
    Returns:
        Formatted research problem
    """
    # Remove extra whitespace
    formatted = " ".join(problem.split())
    
    # Ensure it ends with a question mark if it's a question
    if (formatted.strip().startswith("What") or 
        formatted.strip().startswith("How") or 
        formatted.strip().startswith("Why") or 
        formatted.strip().startswith("Where") or 
        formatted.strip().startswith("When") or 
        formatted.strip().startswith("Who")) and not formatted.strip().endswith("?"):
        formatted = formatted.strip() + "?"
    
    return formatted

def track_research_progress(research_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Track and summarize progress of the research process.
    
    Args:
        research_state: Current state of the research
        
    Returns:
        Summary of research progress
    """
    progress = {
        "problem_defined": research_state.get("problem") is not None,
        "sub_questions_count": len(research_state.get("sub_questions", [])),
        "findings_count": len(research_state.get("findings", [])),
        "citations_count": len(research_state.get("citations", [])),
        "report_generated": research_state.get("report") is not None
    }
    
    # Calculate overall progress as a percentage
    total_steps = 5  # problem, sub-questions, findings, citations, report
    completed_steps = sum([
        progress["problem_defined"],
        progress["sub_questions_count"] > 0,
        progress["findings_count"] > 0,
        progress["citations_count"] > 0,
        progress["report_generated"]
    ])
    
    progress["overall_percentage"] = (completed_steps / total_steps) * 100
    
    return progress

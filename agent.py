# agent.py
"""
Main ResearchAgent implementation that orchestrates the research process.
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import time
import json
import logging
import uuid
import datetime
import traceback
from pathlib import Path

import openai
from openai.agent import Agent, Tool
from openai.error import OpenAIError

from .tools.web_search_tool import WebSearchTool
from .tools.document_analysis_tool import DocumentAnalysisTool
from .tools.information_extraction_tool import InformationExtractionTool
from .tools.citation_generation_tool import CitationGenerationTool
from .tools.report_generation_tool import ReportGenerationTool
from .tools.data_analysis_tool import DataAnalysisTool
from .utils.helpers import format_research_problem, track_research_progress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResearchAgent")

class ResearchError(Exception):
    """Base exception for research-related errors."""
    pass

class APIConfigError(ResearchError):
    """Exception raised for API configuration issues."""
    pass

class ResearchStateError(ResearchError):
    """Exception raised for issues with the research state."""
    pass

class ToolError(ResearchError):
    """Exception raised for issues with research tools."""
    pass

class ResearchAgent:
    """
    The primary agent responsible for orchestrating the entire research process.
    It takes the user's research problem definition as input and generates a comprehensive
    research report with proper citations.
    """
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4-turbo",
        checkpoint_dir: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the ResearchAgent.
        
        Args:
            api_key: OpenAI API key
            model: The model to use for the agent (default: gpt-4-turbo)
            checkpoint_dir: Directory to save research checkpoints (default: None)
            log_level: Logging level (default: INFO)
        
        Raises:
            APIConfigError: If the API key is invalid or not provided
        """
        # Configure logging
        self._configure_logging(log_level)
        
        # Validate API key
        if not api_key:
            logger.error("No API key provided")
            raise APIConfigError("OpenAI API key is required")
        
        # Initialize basic properties
        self.api_key = api_key
        self.model = model
        
        try:
            openai.api_key = api_key
            logger.info(f"Initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI API: {str(e)}")
            raise APIConfigError(f"Failed to initialize OpenAI API: {str(e)}")
        
        # Set up checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"Using checkpoint directory: {checkpoint_dir}")
        
        # Generate a unique session ID for this research instance
        self.session_id = str(uuid.uuid4())
        logger.info(f"Created new research session: {self.session_id}")
        
        # Initialize research tools
        self._init_tools()
        
        # Initialize agent state with metadata
        self._init_research_state()
    
    def _configure_logging(self, log_level: str) -> None:
        """Configure the logger based on the specified level."""
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
        
        # Set the logger level
        logger.setLevel(numeric_level)
    
    def _init_tools(self) -> None:
        """Initialize all research tools and handle potential errors."""
        try:
            # Initialize tools with error handling
            self.web_search_tool = WebSearchTool()
            logger.debug("WebSearchTool initialized")
            
            self.document_analysis_tool = DocumentAnalysisTool()
            logger.debug("DocumentAnalysisTool initialized")
            
            self.information_extraction_tool = InformationExtractionTool()
            logger.debug("InformationExtractionTool initialized")
            
            self.citation_generation_tool = CitationGenerationTool()
            logger.debug("CitationGenerationTool initialized")
            
            self.report_generation_tool = ReportGenerationTool()
            logger.debug("ReportGenerationTool initialized")
            
            self.data_analysis_tool = DataAnalysisTool()
            logger.debug("DataAnalysisTool initialized")
            
            # Create the OpenAI agent tools list
            self.tools = [
                Tool(
                    name="web_search",
                    description="Search the web for information related to the research topic",
                    function=self._safe_tool_call(self.web_search_tool.search, "web_search")
                ),
                Tool(
                    name="document_analysis",
                    description="Analyze and extract text from documents",
                    function=self._safe_tool_call(self.document_analysis_tool.analyze, "document_analysis")
                ),
                Tool(
                    name="information_extraction",
                    description="Extract specific information from text",
                    function=self._safe_tool_call(self.information_extraction_tool.extract, "information_extraction")
                ),
                Tool(
                    name="citation_generation",
                    description="Generate citations for claims based on sources",
                    function=self._safe_tool_call(self.citation_generation_tool.generate_citation, "citation_generation")
                ),
                Tool(
                    name="report_generation",
                    description="Generate a structured research report",
                    function=self._safe_tool_call(self.report_generation_tool.generate_report, "report_generation")
                ),
                Tool(
                    name="data_analysis",
                    description="Analyze numerical or structured data",
                    function=self._safe_tool_call(self.data_analysis_tool.analyze, "data_analysis")
                )
            ]
            logger.info("All research tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {str(e)}")
            self._log_exception("Tool initialization failed")
            raise ToolError(f"Failed to initialize research tools: {str(e)}")
    
    def _safe_tool_call(self, tool_function, tool_name: str):
        """Wrap a tool function with error handling to make it safer."""
        def wrapped_function(*args, **kwargs):
            try:
                start_time = time.time()
                result = tool_function(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log tool usage
                logger.debug(f"Tool {tool_name} executed in {duration:.2f}s")
                
                # Track tool usage in research state
                self._track_tool_usage(tool_name, args, kwargs, duration, success=True)
                
                return result
            except Exception as e:
                logger.error(f"Error in {tool_name}: {str(e)}")
                self._log_exception(f"Tool {tool_name} failed")
                
                # Track failed tool usage
                self._track_tool_usage(tool_name, args, kwargs, 0, success=False, error=str(e))
                
                # Return error information that the agent can understand
                return {
                    "error": f"Tool {tool_name} encountered an error: {str(e)}",
                    "success": False
                }
        
        return wrapped_function
    
    def _track_tool_usage(self, tool_name: str, args: tuple, kwargs: dict, 
                         duration: float, success: bool, error: Optional[str] = None) -> None:
        """Track tool usage for analytics and debugging."""
        # Only track if research state is initialized
        if hasattr(self, "research_state") and self.research_state:
            if "tool_usage" not in self.research_state:
                self.research_state["tool_usage"] = []
            
            # Create tool usage record
            usage_record = {
                "tool": tool_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "duration": duration,
                "success": success
            }
            
            # Add error information if applicable
            if not success and error:
                usage_record["error"] = error
            
            # Add to research state
            self.research_state["tool_usage"].append(usage_record)
    
    def _init_research_state(self) -> None:
        """Initialize or reset the research state."""
        self.research_state = {
            # Core research data
            "problem": None,
            "sub_questions": [],
            "findings": [],
            "citations": [],
            "report": None,
            
            # Metadata
            "session_id": self.session_id,
            "start_time": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
            "status": "initialized",
            "progress": 0,
            "errors": [],
            "tool_usage": [],
            "checkpoints": []
        }
        logger.info("Research state initialized")
    
    def _update_research_state(self, updates: Dict[str, Any]) -> None:
        """
        Update the research state with new information.
        
        Args:
            updates: Dictionary containing the updates to apply
        """
        for key, value in updates.items():
            self.research_state[key] = value
        
        # Update metadata
        self.research_state["last_updated"] = datetime.datetime.now().isoformat()
        
        # Calculate and update progress
        progress = track_research_progress(self.research_state)
        self.research_state["progress"] = progress["overall_percentage"]
        
        logger.debug(f"Research state updated. Progress: {self.research_state['progress']:.1f}%")
    
    def _log_exception(self, context: str) -> None:
        """Log an exception with context and add it to the research state."""
        exc_info = traceback.format_exc()
        logger.error(f"{context}: {exc_info}")
        
        # Add to research state errors
        if hasattr(self, "research_state") and self.research_state:
            if "errors" not in self.research_state:
                self.research_state["errors"] = []
            
            self.research_state["errors"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "context": context,
                "traceback": exc_info
            })
    
    def create_agent(self) -> Agent:
        """
        Create and return an OpenAI Agent instance with research capabilities.
        
        Returns:
            An OpenAI Agent instance configured for research
            
        Raises:
            APIConfigError: If there's an issue creating the agent
        """
        try:
            instructions = """
            You are an advanced research agent capable of conducting thorough research on any topic.
            Your goal is to create well-researched, accurate reports with proper citations.
            
            Follow these steps for any research task:
            1. Analyze the research problem and break it down into sub-questions
            2. For each sub-question, search for relevant information using web_search
            3. Analyze documents and extract key information using document_analysis and information_extraction
            4. For each claim or piece of information, generate a proper citation using citation_generation
            5. Synthesize findings into a comprehensive report using report_generation
            6. If you encounter data that needs analysis, use data_analysis
            
            Always maintain academic integrity by:
            - Citing all sources properly
            - Distinguishing between facts and opinions
            - Presenting balanced perspectives on controversial topics
            - Verifying information across multiple sources when possible
            
            If you encounter any errors while using tools, try an alternative approach or skip to the next part of your research if necessary. Document any limitations in your final report.
            """
            
            # Create agent
            agent = Agent(
                model=self.model,
                tools=self.tools,
                instructions=instructions
            )
            
            logger.info(f"Created agent with model {self.model} and {len(self.tools)} tools")
            return agent
            
        except OpenAIError as e:
            logger.error(f"OpenAI API error creating agent: {str(e)}")
            self._log_exception("Failed to create agent")
            raise APIConfigError(f"Failed to create OpenAI agent: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error creating agent: {str(e)}")
            self._log_exception("Failed to create agent")
            raise ResearchError(f"Failed to create agent: {str(e)}")
    
    def conduct_research(self, 
                        research_problem: str, 
                        citation_style: str = "APA",
                        max_iterations: int = 1,
                        timeout: Optional[int] = None,
                        checkpoint_interval: int = 300) -> Dict[str, Any]:
        """
        Conduct research on the given problem and generate a report with citations.
        
        Args:
            research_problem: The research problem to investigate
            citation_style: The citation style to use (APA, MLA, Chicago, etc.)
            max_iterations: Maximum number of research iterations (default: 1)
            timeout: Optional timeout in seconds for the whole research process
            checkpoint_interval: How often to save checkpoints in seconds (default: 300)
            
        Returns:
            A dictionary containing the research report and metadata
            
        Raises:
            ResearchError: If there's an issue during the research process
        """
        # Reset research state
        self._init_research_state()
        
        try:
            # Format the research problem
            formatted_problem = format_research_problem(research_problem)
            self._update_research_state({
                "problem": formatted_problem,
                "status": "researching",
                "citation_style": citation_style
            })
            
            logger.info(f"Starting research on problem: {formatted_problem}")
            
            # Configure citation style
            self.citation_generation_tool.set_citation_style(citation_style)
            
            # Create the agent
            agent = self.create_agent()
            
            # Set up checkpoint timer
            last_checkpoint_time = time.time()
            start_time = time.time()
            
            # Starting the research process
            initial_message = f"""
            I need to conduct thorough research on the following topic:
            
            {formatted_problem}
            
            Please help me create a comprehensive research report with proper {citation_style} citations.
            Begin by breaking down this problem into manageable sub-questions for investigation.
            """
            
            # Track research iterations
            iterations = 0
            current_message = initial_message
            
            # Main research loop
            while iterations < max_iterations:
                iterations += 1
                logger.info(f"Starting research iteration {iterations}/{max_iterations}")
                
                # Check timeout
                if timeout and (time.time() - start_time > timeout):
                    logger.warning(f"Research timed out after {timeout} seconds")
                    self._update_research_state({"status": "timeout"})
                    break
                
                # Run the agent
                try:
                    result = agent.run(current_message)
                    
                    # Process the agent's output and update the research state
                    self._process_agent_output(result)
                    
                    # If this isn't the last iteration, prepare for the next iteration
                    if iterations < max_iterations:
                        # Generate a follow-up message based on current state
                        current_message = self._generate_followup_message()
                    
                    logger.info(f"Completed research iteration {iterations}")
                    
                except OpenAIError as e:
                    logger.error(f"OpenAI API error: {str(e)}")
                    self._log_exception(f"Error in research iteration {iterations}")
                    self._update_research_state({"status": "error"})
                    raise ResearchError(f"OpenAI API error during research: {str(e)}")
                
                # Check if it's time to save a checkpoint
                current_time = time.time()
                if self.checkpoint_dir and (current_time - last_checkpoint_time > checkpoint_interval):
                    self._save_checkpoint()
                    last_checkpoint_time = current_time
            
            # Final checkpoint
            if self.checkpoint_dir:
                self._save_checkpoint(is_final=True)
                
            # Mark research as completed
            self._update_research_state({"status": "completed"})
            logger.info(f"Research completed in {iterations} iterations")
            
            return {
                "problem": self.research_state["problem"],
                "report": self.research_state["report"],
                "citations": self.research_state["citations"],
                "sub_questions": self.research_state["sub_questions"],
                "findings": self.research_state["findings"],
                "metadata": {
                    "session_id": self.session_id,
                    "start_time": self.research_state["start_time"],
                    "end_time": datetime.datetime.now().isoformat(),
                    "duration": time.time() - start_time,
                    "iterations": iterations,
                    "status": self.research_state["status"],
                    "progress": self.research_state["progress"]
                }
            }
            
        except Exception as e:
            logger.error(f"Research error: {str(e)}")
            self._log_exception("Error in conduct_research")
            self._update_research_state({"status": "error"})
            
            # Try to save a checkpoint if there's an error
            if self.checkpoint_dir:
                self._save_checkpoint(is_error=True)
                
            raise ResearchError(f"Error during research process: {str(e)}")
    
    def _generate_followup_message(self) -> str:
        """Generate a follow-up message for the next research iteration based on current state."""
        # Analyze current research state to identify gaps
        progress = track_research_progress(self.research_state)
        
        # Check what's missing
        if not self.research_state["sub_questions"]:
            return "Please break down the research problem into specific sub-questions that we need to investigate."
        
        if not self.research_state["findings"]:
            return f"So far, you've identified these sub-questions: {', '.join(self.research_state['sub_questions'])}. Please research them and collect findings."
        
        if not self.research_state["citations"]:
            return "You've gathered some findings, but they need proper citations. Please ensure all claims are properly cited."
        
        if not self.research_state["report"]:
            return "Please synthesize all findings into a comprehensive research report with proper citations."
        
        # Default: ask for improvements
        return "Please review the current research and identify any gaps or areas that need more depth. Then improve the report accordingly."
    
    def _process_agent_output(self, result: Dict[str, Any]) -> None:
        """
        Process the output from the agent and update the research state.
        
        Args:
            result: The result dictionary from the agent
        """
        updates = {}
        
        # Extract the final report
        if "report" in result:
            updates["report"] = result["report"]
        elif "messages" in result and result["messages"]:
            # If the agent didn't explicitly return a report, use the last message
            updates["report"] = result["messages"][-1]["content"]
        
        # Extract citations if available
        if "citations" in result:
            updates["citations"] = result["citations"]
        
        # Extract sub-questions if available
        if "sub_questions" in result:
            updates["sub_questions"] = result["sub_questions"]
        elif not self.research_state["sub_questions"]:
            # Try to extract sub-questions from the text if not explicitly provided
            extracted_questions = self._extract_questions_from_text(updates.get("report", ""))
            if extracted_questions:
                updates["sub_questions"] = extracted_questions
        
        # Extract findings if available
        if "findings" in result:
            updates["findings"] = result["findings"]
        
        # Update the research state
        self._update_research_state(updates)
    
    def _extract_questions_from_text(self, text: str) -> List[str]:
        """Extract potential research questions from text content."""
        questions = []
        
        # Look for numbered or bullet lists with questions
        import re
        patterns = [
            r'\d+\.\s+([A-Z][^.?!]*\?)',  # Numbered questions
            r'[-•*]\s+([A-Z][^.?!]*\?)',  # Bulleted questions
            r'([A-Z][^.?!]*\?)'  # Any question that starts with capital letter
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Clean up the question
                question = match.strip()
                if question and len(question) > 10:  # Ensure it's reasonably long
                    questions.append(question)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_questions = [q for q in questions if not (q in seen or seen.add(q))]
        
        return unique_questions[:10]  # Limit to 10 questions
    
    def _save_checkpoint(self, is_final: bool = False, is_error: bool = False) -> None:
        """
        Save a checkpoint of the current research state.
        
        Args:
            is_final: Whether this is the final checkpoint
            is_error: Whether this checkpoint is being saved after an error
        """
        if not self.checkpoint_dir:
            return
        
        try:
            # Generate checkpoint filename
            checkpoint_type = "final" if is_final else "error" if is_error else "checkpoint"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.session_id}_{checkpoint_type}_{timestamp}.json"
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            # Add checkpoint to research state
            checkpoint_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "filename": filename,
                "type": checkpoint_type
            }
            self.research_state["checkpoints"].append(checkpoint_info)
            
            # Save the checkpoint
            with open(filepath, "w") as f:
                json.dump(self.research_state, f, indent=2)
                
            logger.info(f"Saved {checkpoint_type} checkpoint to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            self._log_exception("Checkpoint save failed")
    
    def save_report(self, file_path: str) -> None:
        """
        Save the research report to a file.
        
        Args:
            file_path: Path to save the report to
            
        Raises:
            ResearchStateError: If no report is available
            IOError: If there's an issue writing the file
        """
        if not self.research_state["report"]:
            logger.error("No research report available to save")
            raise ResearchStateError("No research report available to save")
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, "w") as f:
                f.write(self.research_state["report"])
                
            logger.info(f"Report saved to {file_path}")
            
        except IOError as e:
            logger.error(f"Failed to save report: {str(e)}")
            self._log_exception(f"Save report failed: {file_path}")
            raise
    
    def save_research_data(self, file_path: str) -> None:
        """
        Save all research data including citations to a JSON file.
        
        Args:
            file_path: Path to save the research data to
            
        Raises:
            ResearchStateError: If no research data is available
            IOError: If there's an issue writing the file
        """
        if not self.research_state["report"]:
            logger.error("No research data available to save")
            raise ResearchStateError("No research data available to save")
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Create a clean copy for saving (without potentially large internal state)
            save_data = {
                "problem": self.research_state["problem"],
                "sub_questions": self.research_state["sub_questions"],
                "findings": self.research_state["findings"],
                "citations": self.research_state["citations"],
                "report": self.research_state["report"],
                "metadata": {
                    "session_id": self.session_id,
                    "start_time": self.research_state["start_time"],
                    "end_time": datetime.datetime.now().isoformat(),
                    "status": self.research_state["status"],
                    "progress": self.research_state["progress"]
                }
            }
            
            with open(file_path, "w") as f:
                json.dump(save_data, f, indent=2)
                
            logger.info(f"Research data saved to {file_path}")
            
        except IOError as e:
            logger.error(f"Failed to save research data: {str(e)}")
            self._log_exception(f"Save research data failed: {file_path}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a research checkpoint to resume a previous session.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            The loaded research state
            
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
            ResearchStateError: If there's an issue with the checkpoint data
        """
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
                
            # Validate the checkpoint data
            required_keys = ["problem", "session_id", "status"]
            if not all(key in checkpoint_data for key in required_keys):
                logger.error("Invalid checkpoint data: missing required keys")
                raise ResearchStateError("Invalid checkpoint data: missing required keys")
            
            # Update session ID to match the loaded checkpoint
            self.session_id = checkpoint_data["session_id"]
            
            # Replace current research state with checkpoint data
            self.research_state = checkpoint_data
            
            # Update the 'last_updated' timestamp
            self.research_state["last_updated"] = datetime.datetime.now().isoformat()
            
            # Add resume information
            if "resume_history" not in self.research_state:
                self.research_state["resume_history"] = []
                
            self.research_state["resume_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "checkpoint_path": checkpoint_path
            })
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(f"Resumed session: {self.session_id}")
            
            return self.research_state
            
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON in checkpoint file")
            self._log_exception(f"Failed to load checkpoint: {checkpoint_path}")
            raise ResearchStateError("Invalid JSON in checkpoint file")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            self._log_exception(f"Failed to load checkpoint: {checkpoint_path}")
            raise ResearchStateError(f"Error loading checkpoint: {str(e)}")
    
    def get_research_status(self) -> Dict[str, Any]:
        """
        Get the current status and progress of the research.
        
        Returns:
            Dictionary with status information
        """
        # Calculate detailed progress
        progress = track_research_progress(self.research_state)
        
        # Return status information
        return {
            "session_id": self.session_id,
            "status": self.research_state["status"],
            "progress": {
                "overall": progress["overall_percentage"],
                "problem_defined": progress["problem_defined"],
                "sub_questions_count": progress["sub_questions_count"],
                "findings_count": progress["findings_count"],
                "citations_count": progress["citations_count"],
                "report_generated": progress["report_generated"]
            },
            "start_time": self.research_state["start_time"],
            "last_updated": self.research_state["last_updated"],
            "error_count": len(self.research_state.get("errors", [])),
            "checkpoint_count": len(self.research_state.get("checkpoints", [])),
            "tool_usage_count": len(self.research_state.get("tool_usage", []))
        }
    
    def get_error_report(self) -> List[Dict[str, Any]]:
        """
        Get a list of all errors that occurred during the research process.
        
        Returns:
            List of error details
        """
        return self.research_state.get("errors", [])
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about tool usage during the research process.
        
        Returns:
            Dictionary with tool usage statistics
        """
        stats = {
            "total_calls": 0,
            "success_rate": 0,
            "tools": {},
            "average_duration": 0
        }
        
        tool_usage = self.research_state.get("tool_usage", [])
        if not tool_usage:
            return stats
        
        # Calculate overall stats
        stats["total_calls"] = len(tool_usage)
        successful_calls = sum(1 for usage in tool_usage if usage.get("success", False))
        stats["success_rate"] = (successful_calls / len(tool_usage)) * 100 if tool_usage else 0
        
        # Calculate average duration of successful calls
        successful_durations = [usage.get("duration", 0) for usage in tool_usage if usage.get("success", False)]
        stats["average_duration"] = sum(successful_durations) / len(successful_durations) if successful_durations else 0
        
        # Calculate per-tool stats
        tool_calls = {}
        for usage in tool_usage:
            tool_name = usage.get("tool", "unknown")
            if tool_name not in tool_calls:
                tool_calls[tool_name] = {
                    "calls": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_duration": 0
                }
            
            tool_calls[tool_name]["calls"] += 1
            if usage.get("success", False):
                tool_calls[tool_name]["successful"] += 1
                tool_calls[tool_name]["total_duration"] += usage.get("duration", 0)
            else:
                tool_calls[tool_name]["failed"] += 1
        
        # Calculate success rates and average durations per tool
        for tool_name, data in tool_calls.items():
            stats["tools"][tool_name] = {
                "calls": data["calls"],
                "success_rate": (data["successful"] / data["calls"]) * 100 if data["calls"] > 0 else 0,
                "average_duration": data["total_duration"] / data["successful"] if data["successful"] > 0 else 0
            }
        
        return stats
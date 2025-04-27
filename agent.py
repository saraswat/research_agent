# agent.py
"""
Main ResearchAgent implementation that orchestrates the research process.
"""
from typing import List, Dict, Any, Optional, Union, Tuple, Literal
import os
import time
import json
import logging
import uuid
import datetime
import traceback
import re
from pathlib import Path

# Import backends (conditionally to handle cases where one might not be installed)
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    from openai.types.chat import ChatCompletion
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
    
class GeminiError(ResearchError):
    """Exception raised for issues with the Gemini API."""
    pass
    
class OpenAIError(ResearchError):
    """Exception raised for issues with the OpenAI API."""
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
        backend: Literal["gemini", "openai"] = "gemini",
        model: str = None,
        checkpoint_dir: Optional[str] = None,
        log_level: str = "INFO",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 4096
    ):
        """
        Initialize the ResearchAgent.
        
        Args:
            api_key: API key for the selected backend (Google or OpenAI)
            backend: The LLM backend to use: "gemini" or "openai" (default: "gemini")
            model: The model to use for the agent (default depends on backend)
            checkpoint_dir: Directory to save research checkpoints (default: None)
            log_level: Logging level (default: INFO)
            temperature: Controls randomness in response generation (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.95)
            top_k: Top-k sampling parameter (default: 40, only used for Gemini)
            max_tokens: Maximum tokens in response (default: 4096)
        
        Raises:
            APIConfigError: If the API key is invalid or not provided
            ImportError: If the selected backend is not available
        """
        # Configure logging
        self._configure_logging(log_level)
        
        # Validate API key
        if not api_key:
            logger.error("No API key provided")
            raise APIConfigError("API key is required")
        
        # Initialize basic properties
        self.api_key = api_key
        self.backend = backend
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        
        # Set default model based on backend if not provided
        if model is None:
            if backend == "gemini":
                self.model = "gemini-1.5-pro"
            elif backend == "openai":
                self.model = "gpt-4o"
            else:
                raise APIConfigError(f"Unknown backend: {backend}")
        else:
            self.model = model
            
        # Initialize the specified backend
        if backend == "gemini":
            self._init_gemini_backend()
        elif backend == "openai":
            self._init_openai_backend()
        else:
            logger.error(f"Unsupported backend: {backend}")
            raise APIConfigError(f"Unsupported backend: {backend}. Choose 'gemini' or 'openai'.")
        
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
        
    def _init_gemini_backend(self):
        """Initialize the Gemini backend."""
        if not GEMINI_AVAILABLE:
            logger.error("Google Gemini package not installed")
            raise ImportError("To use Gemini backend, install google-generativeai package")
            
        try:
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            
            # Set default generation configuration
            self.generation_config = GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_tokens,
            )
            
            # Verify model is available
            models = genai.list_models()
            available_models = [m.name for m in models]
            # Extract just the model name without the full path
            model_short_name = self.model.split('/')[-1] if '/' in self.model else self.model
            
            if not any(model_short_name in m for m in available_models):
                available_gemini_models = [m for m in available_models if 'gemini' in m]
                logger.warning(f"Model {self.model} not found in available models. Available Gemini models: {available_gemini_models}")
                if available_gemini_models:
                    self.model = available_gemini_models[0]
                    logger.info(f"Using alternative model: {self.model}")
                else:
                    raise APIConfigError(f"No Gemini models available")
            
            logger.info(f"Initialized with Gemini model: {self.model}")
            
            # Initialize the model
            self.gemini_model = genai.GenerativeModel(model_name=self.model, 
                                                     generation_config=self.generation_config)
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise APIConfigError(f"Failed to initialize Gemini API: {str(e)}")
            
    def _init_openai_backend(self):
        """Initialize the OpenAI backend."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI package not installed")
            raise ImportError("To use OpenAI backend, install openai package")
            
        try:
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=self.api_key)
            
            # Initialize OpenAI models list to validate model name
            available_models = []
            try:
                models_response = self.openai_client.models.list()
                available_models = [model.id for model in models_response.data]
            except Exception as e:
                logger.warning(f"Could not list OpenAI models: {str(e)}")
                
            # Check if model exists or use default
            if available_models and self.model not in available_models:
                # Try to find a suitable alternative
                if "gpt-4" in available_models:
                    self.model = "gpt-4"
                elif "gpt-3.5-turbo" in available_models:
                    self.model = "gpt-3.5-turbo"
                    
                logger.warning(f"Selected model not found, using alternative: {self.model}")
                
            logger.info(f"Initialized with OpenAI model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI API: {str(e)}")
            raise APIConfigError(f"Failed to initialize OpenAI API: {str(e)}")
    
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
            
            # Define tool schemas for both backends
            self.tool_schemas = {
                "WebSearch": {
                    "name": "WebSearch",
                    "description": "Search the web for current information on a topic",
                    "function": self.web_search_tool.search,
                    "parameters": {
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                            "num_results": {"type": "integer", "description": "Number of results to return"}
                        },
                        "required": ["query"]
                    }
                },
                "DocumentAnalysis": {
                    "name": "DocumentAnalysis",
                    "description": "Analyze a document URL or file path",
                    "function": self.document_analysis_tool.analyze,
                    "parameters": {
                        "properties": {
                            "document_source": {"type": "string", "description": "The URL or file path of the document"},
                            "document_type": {"type": "string", "description": "The document type (optional)"}
                        },
                        "required": ["document_source"]
                    }
                },
                "InformationExtraction": {
                    "name": "InformationExtraction",
                    "description": "Extract specific information from text",
                    "function": self.information_extraction_tool.extract,
                    "parameters": {
                        "properties": {
                            "text": {"type": "string", "description": "The text to extract information from"},
                            "instructions": {"type": "string", "description": "Instructions on what information to extract"},
                            "source": {"type": "string", "description": "Optional source reference for citation purposes"},
                            "fact_check": {"type": "boolean", "description": "Whether to perform fact checking"}
                        },
                        "required": ["text", "instructions"]
                    }
                },
                "CitationGeneration": {
                    "name": "CitationGeneration",
                    "description": "Generate a citation for a claim based on a source",
                    "function": self.citation_generation_tool.generate_citation,
                    "parameters": {
                        "properties": {
                            "claim": {"type": "string", "description": "The claim to cite"},
                            "source": {"type": "object", "description": "The source information for the citation"}
                        },
                        "required": ["claim", "source"]
                    }
                },
                "DataAnalysis": {
                    "name": "DataAnalysis",
                    "description": "Analyze numerical or structured data",
                    "function": self.data_analysis_tool.analyze,
                    "parameters": {
                        "properties": {
                            "data": {"type": "object", "description": "The data to analyze"},
                            "analysis_type": {"type": "string", "description": "The type of analysis to perform"},
                            "parameters": {"type": "object", "description": "Optional parameters for the analysis"}
                        },
                        "required": ["data", "analysis_type"]
                    }
                },
                "ReportGeneration": {
                    "name": "ReportGeneration",
                    "description": "Generate a structured research report",
                    "function": self.report_generation_tool.generate_report,
                    "parameters": {
                        "properties": {
                            "title": {"type": "string", "description": "The report title"},
                            "data": {"type": "object", "description": "The research data to include in the report"},
                            "format": {"type": "string", "description": "The output format (markdown, html, etc.)"},
                            "style": {"type": "string", "description": "The citation style to use"}
                        },
                        "required": ["title", "data"]
                    }
                }
            }
            
            # Set up tools based on the backend
            if self.backend == "openai":
                # Set up OpenAI-compatible tools
                self._init_openai_tools()
            else:
                # Gemini tools are defined in the tool_schemas dict
                self.tools = self.tool_schemas
            
            logger.info(f"All research tools initialized successfully for {self.backend} backend")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {str(e)}")
            self._log_exception("Tool initialization failed")
            raise ToolError(f"Failed to initialize research tools: {str(e)}")
            
    def _init_openai_tools(self) -> None:
        """Initialize tools for the OpenAI backend."""
        self.openai_tools = []
        
        for tool_id, tool_info in self.tool_schemas.items():
            # Convert our tool schema to OpenAI format
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_id.lower(),
                    "description": tool_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool_info["parameters"]["properties"],
                        "required": tool_info["parameters"]["required"]
                    }
                }
            }
            self.openai_tools.append(openai_tool)
            
        logger.debug(f"Initialized {len(self.openai_tools)} OpenAI tools")
    
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
    
    def create_agent(self):
        """
        Set up the research agent with appropriate tools and configuration.
        
        Returns:
            The configured model (either Gemini or OpenAI)
            
        Raises:
            APIConfigError: If there's an issue creating the agent
        """
        try:
            instructions = """
            You are an advanced research agent capable of conducting thorough research on any topic.
            Your goal is to create well-researched, accurate reports with proper citations.
            
            Follow these steps for any research task:
            1. Analyze the research problem and break it down into sub-questions
            2. For each sub-question, search for relevant information using WebSearch
            3. Analyze documents and extract key information using DocumentAnalysis and InformationExtraction
            4. For each claim or piece of information, generate a proper citation using CitationGeneration
            5. Synthesize findings into a comprehensive report
            6. If you encounter data that needs analysis, use DataAnalysis
            
            Always maintain academic integrity by:
            - Citing all sources properly
            - Distinguishing between facts and opinions
            - Presenting balanced perspectives on controversial topics
            - Verifying information across multiple sources when possible
            
            If you encounter any errors while using tools, try an alternative approach or skip to the next part of your research if necessary. Document any limitations in your final report.
            """
            
            # Create the agent based on selected backend
            if self.backend == "gemini":
                return self._create_gemini_agent(instructions)
            elif self.backend == "openai":
                return self._create_openai_agent(instructions)
            else:
                raise APIConfigError(f"Unknown backend: {self.backend}")
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            self._log_exception("Failed to create agent")
            raise APIConfigError(f"Failed to create agent: {str(e)}")
            
    def _create_gemini_agent(self, instructions: str):
        """
        Create a Gemini-based research agent.
        
        Args:
            instructions: System instructions for the agent
            
        Returns:
            The configured Gemini model
        """
        try:
            # The tools were already initialized in _init_tools
            
            # Update the model with the system instructions
            self.gemini_model = genai.GenerativeModel(
                model_name=self.model,
                generation_config=self.generation_config,
                system_instruction=instructions
            )
            
            logger.info(f"Created agent with Gemini model {self.model} and {len(self.tools)} tools")
            return self.gemini_model
            
        except Exception as e:
            logger.error(f"Error creating Gemini agent: {str(e)}")
            self._log_exception("Failed to create Gemini agent")
            raise GeminiError(f"Failed to create Gemini agent: {str(e)}")
            
    def _create_openai_agent(self, instructions: str):
        """
        Create an OpenAI-based research agent.
        
        Args:
            instructions: System instructions for the agent
            
        Returns:
            The OpenAI client
        """
        try:
            # The OpenAI tools were already initialized in _init_openai_tools
            
            # Store the system instructions for later use
            self.openai_system_instructions = instructions
            
            logger.info(f"Created agent with OpenAI model {self.model} and {len(self.openai_tools)} tools")
            return self.openai_client
            
        except Exception as e:
            logger.error(f"Error creating OpenAI agent: {str(e)}")
            self._log_exception("Failed to create OpenAI agent")
            raise OpenAIError(f"Failed to create OpenAI agent: {str(e)}")
    
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
                
                # Run the agent based on the selected backend
                try:
                    # Query the selected backend
                    if self.backend == "gemini":
                        result = self._query_gemini(current_message)
                    elif self.backend == "openai":
                        result = self._query_openai(current_message)
                    else:
                        raise APIConfigError(f"Unknown backend: {self.backend}")
                    
                    # Process the agent's output and update the research state
                    self._process_agent_output(result)
                    
                    # If this isn't the last iteration, prepare for the next iteration
                    if iterations < max_iterations:
                        # Generate a follow-up message based on current state
                        current_message = self._generate_followup_message()
                    
                    logger.info(f"Completed research iteration {iterations}")
                    
                except Exception as e:
                    logger.error(f"{self.backend.capitalize()} API error: {str(e)}")
                    self._log_exception(f"Error in research iteration {iterations}")
                    self._update_research_state({"status": "error"})
                    raise ResearchError(f"{self.backend.capitalize()} API error during research: {str(e)}")
                
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
    
    def _query_gemini(self, user_message: str) -> Dict[str, Any]:
        """
        Query the Gemini model and process any tool calls.
        
        Args:
            user_message: The user's message to send to Gemini
            
        Returns:
            A dictionary containing the model's response and any tool outputs
            
        Raises:
            GeminiError: If there's an issue with the Gemini API
        """
        try:
            logger.info("Sending query to Gemini")
            
            # Start a chat session
            chat = self.gemini_model.start_chat(history=[])
            
            # Get the model's response
            response = chat.send_message(user_message)
            
            # Process tool calls if the model requests them
            result = {
                "messages": [],
                "report": response.text
            }
            
            # Check for potential tool calls in the response
            # Gemini doesn't have a standardized tool calling format like OpenAI,
            # so we need to parse the response text to identify tool requests
            
            # Parse the response for tool calls - we're looking for patterns like:
            # "I'll use the WebSearch tool to find information about..."
            # "Let me use InformationExtraction to analyze..."
            tool_patterns = [
                (r'use (?:the )?(WebSearch)(.*?)(?:to|:)(.*?)(?:\.|$)', "WebSearch", ["query"]),
                (r'use (?:the )?(DocumentAnalysis)(.*?)(?:to|:)(.*?)(?:\.|$)', "DocumentAnalysis", ["document_source"]),
                (r'use (?:the )?(InformationExtraction)(.*?)(?:to|:)(.*?)(?:\.|$)', "InformationExtraction", ["text", "instructions"]),
                (r'use (?:the )?(CitationGeneration)(.*?)(?:to|:)(.*?)(?:\.|$)', "CitationGeneration", ["claim", "source"]),
                (r'use (?:the )?(DataAnalysis)(.*?)(?:to|:)(.*?)(?:\.|$)', "DataAnalysis", ["data", "analysis_type"])
            ]
            
            # Track all tool outputs
            tool_outputs = []
            
            # Extract and execute tool calls
            for pattern, tool_name, required_params in tool_patterns:
                matches = re.finditer(pattern, response.text, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    # Extract the tool name and potential arguments
                    tool_args_text = match.group(3).strip()
                    
                    # Parse arguments (this is a simplified approach)
                    tool_args = self._parse_tool_args(tool_args_text, tool_name)
                    
                    # Check if required parameters are present
                    if all(param in tool_args for param in required_params):
                        # Execute the tool
                        try:
                            tool_result = self.tools[tool_name]["function"](**tool_args)
                            
                            # Log tool usage
                            self._log_tool_usage(tool_name, tool_args, True, tool_result)
                            
                            # Add tool output to results
                            tool_outputs.append({
                                "tool_name": tool_name,
                                "args": tool_args,
                                "result": tool_result
                            })
                            
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {str(e)}")
                            self._log_tool_usage(tool_name, tool_args, False, str(e))
                            
                            # Add error information
                            tool_outputs.append({
                                "tool_name": tool_name,
                                "args": tool_args,
                                "error": str(e)
                            })
            
            # If we have tool outputs, send a follow-up message with the results
            if tool_outputs:
                tool_results_text = self._format_tool_results(tool_outputs)
                
                # Send the tool results back to the model
                follow_up_response = chat.send_message(tool_results_text)
                
                # Update the result with the final response
                result["messages"] = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response.text},
                    {"role": "user", "content": tool_results_text},
                    {"role": "assistant", "content": follow_up_response.text}
                ]
                result["report"] = follow_up_response.text
                result["tool_outputs"] = tool_outputs
            else:
                # No tool calls, just return the response
                result["messages"] = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response.text}
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying Gemini: {str(e)}")
            self._log_exception("Gemini query failed")
            raise GeminiError(f"Error querying Gemini: {str(e)}")
            
    def _query_openai(self, user_message: str) -> Dict[str, Any]:
        """
        Query the OpenAI model and process any tool calls.
        
        Args:
            user_message: The user's message to send to OpenAI
            
        Returns:
            A dictionary containing the model's response and any tool outputs
            
        Raises:
            OpenAIError: If there's an issue with the OpenAI API
        """
        try:
            logger.info("Sending query to OpenAI")
            
            # Initialize result structure
            result = {
                "messages": [],
                "report": "",
                "tool_outputs": []
            }
            
            # Prepare the message with system instructions
            messages = [
                {"role": "system", "content": self.openai_system_instructions},
                {"role": "user", "content": user_message}
            ]
            
            # Create the OpenAI chat completion with tools
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.openai_tools,
                tool_choice="auto",  # Let the model decide when to use tools
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            
            # Extract the response content
            first_response = response.choices[0].message
            
            # Track the conversation
            result["messages"].append({"role": "user", "content": user_message})
            result["messages"].append({"role": "assistant", "content": first_response.content or ""})
            
            # Process any tool calls
            if hasattr(first_response, 'tool_calls') and first_response.tool_calls:
                # Execute each tool call
                tool_outputs = []
                tool_call_messages = [first_response]
                
                for tool_call in first_response.tool_calls:
                    # Parse the function call
                    function_name = tool_call.function.name
                    tool_name = function_name.capitalize()  # Convert back to our format
                    
                    try:
                        # Parse arguments
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Execute the tool if it exists in our tools
                        if tool_name in self.tool_schemas:
                            tool_result = self.tool_schemas[tool_name]["function"](**function_args)
                            
                            # Log the successful tool usage
                            self._log_tool_usage(tool_name, function_args, True, tool_result)
                            
                            # Add to tool outputs
                            tool_outputs.append({
                                "tool_name": tool_name,
                                "tool_call_id": tool_call.id,
                                "args": function_args,
                                "result": tool_result
                            })
                        else:
                            # Tool not found
                            error_msg = f"Tool {tool_name} not found"
                            logger.error(error_msg)
                            self._log_tool_usage(tool_name, function_args, False, error_msg)
                            
                            tool_outputs.append({
                                "tool_name": tool_name,
                                "tool_call_id": tool_call.id,
                                "args": function_args,
                                "error": error_msg
                            })
                    
                    except Exception as e:
                        # Error executing tool
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        logger.error(error_msg)
                        self._log_tool_usage(tool_name, {}, False, error_msg)
                        
                        tool_outputs.append({
                            "tool_name": tool_name,
                            "tool_call_id": tool_call.id,
                            "error": error_msg
                        })
                
                # Prepare tool responses for the follow-up message
                for tool_output in tool_outputs:
                    if "error" in tool_output:
                        # Error case
                        tool_response = {
                            "tool_call_id": tool_output["tool_call_id"],
                            "role": "tool",
                            "name": tool_output["tool_name"].lower(),
                            "content": f"Error: {tool_output['error']}"
                        }
                    else:
                        # Success case - format the result as JSON
                        result_json = json.dumps(tool_output["result"])
                        tool_response = {
                            "tool_call_id": tool_output["tool_call_id"],
                            "role": "tool",
                            "name": tool_output["tool_name"].lower(),
                            "content": result_json
                        }
                    
                    # Add to messages
                    tool_call_messages.append(tool_response)
                
                # Get a follow-up response from the model with the tool results
                if tool_outputs:
                    # Prepare the full conversation history
                    follow_up_messages = [
                        {"role": "system", "content": self.openai_system_instructions},
                        {"role": "user", "content": user_message},
                        *tool_call_messages
                    ]
                    
                    # Get the follow-up response
                    follow_up_response = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=follow_up_messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens
                    )
                    
                    final_response = follow_up_response.choices[0].message
                    
                    # Update the result
                    result["report"] = final_response.content
                    result["messages"].append({"role": "assistant", "content": final_response.content})
                    result["tool_outputs"] = tool_outputs
                else:
                    # No tool outputs were created, just use the initial response
                    result["report"] = first_response.content or ""
            else:
                # No tool calls in the response
                result["report"] = first_response.content or ""
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying OpenAI: {str(e)}")
            self._log_exception("OpenAI query failed")
            raise OpenAIError(f"Error querying OpenAI: {str(e)}")
            
    def _parse_tool_args(self, args_text: str, tool_name: str) -> Dict[str, Any]:
        """
        Parse tool arguments from text.
        
        Args:
            args_text: The text containing the arguments
            tool_name: The name of the tool
            
        Returns:
            Dictionary of parsed arguments
        """
        tool_args = {}
        
        # Different parsing logic based on tool type
        if tool_name == "WebSearch":
            # Simple case - the text is likely just the query
            tool_args["query"] = args_text.strip()
            # Default number of results
            tool_args["num_results"] = 5
            
        elif tool_name == "DocumentAnalysis":
            # Look for URLs or file paths
            url_match = re.search(r'https?://[^\s"\']+', args_text)
            if url_match:
                tool_args["document_source"] = url_match.group(0)
            else:
                # Use the whole text as document source
                tool_args["document_source"] = args_text.strip()
                
        elif tool_name == "InformationExtraction":
            # This is more complex, needs text and instructions
            # Try to identify instructions first
            instruction_match = re.search(r'instructions?[:\s]+([^.]+)', args_text, re.IGNORECASE)
            if instruction_match:
                instructions = instruction_match.group(1).strip()
                # The rest is probably the text
                text = args_text.replace(instruction_match.group(0), "").strip()
                tool_args["instructions"] = instructions
                tool_args["text"] = text
            else:
                # Simple splitting - first part is text, second is instructions
                parts = args_text.split(".", 1)
                if len(parts) > 1:
                    tool_args["text"] = parts[0].strip()
                    tool_args["instructions"] = parts[1].strip()
                else:
                    # Can't clearly identify, use defaults
                    tool_args["text"] = args_text.strip()
                    tool_args["instructions"] = "Extract key information"
                    
        elif tool_name == "CitationGeneration":
            # Look for claim and source
            claim_match = re.search(r'claim[:\s]+([^.]+)', args_text, re.IGNORECASE)
            source_match = re.search(r'source[:\s]+([^.]+)', args_text, re.IGNORECASE)
            
            if claim_match:
                tool_args["claim"] = claim_match.group(1).strip()
            else:
                tool_args["claim"] = args_text.strip()
                
            if source_match:
                # Try to parse source as JSON if it looks like it
                source_text = source_match.group(1).strip()
                if source_text.startswith("{") and source_text.endswith("}"):
                    try:
                        source = json.loads(source_text)
                        tool_args["source"] = source
                    except:
                        # If not valid JSON, use as string
                        tool_args["source"] = {"text": source_text}
                else:
                    tool_args["source"] = {"text": source_text}
            else:
                tool_args["source"] = {"text": "Unknown source"}
        
        elif tool_name == "DataAnalysis":
            # Look for data and analysis type
            analysis_match = re.search(r'analysis[:\s]+([^.]+)', args_text, re.IGNORECASE)
            if analysis_match:
                tool_args["analysis_type"] = analysis_match.group(1).strip()
            else:
                tool_args["analysis_type"] = "descriptive_statistics"
                
            # For data, try to find structured data or just use the text
            tool_args["data"] = {"text": args_text.strip()}
            
        return tool_args
        
    def _format_tool_results(self, tool_outputs: List[Dict[str, Any]]) -> str:
        """
        Format tool results for sending back to the model.
        
        Args:
            tool_outputs: List of tool outputs
            
        Returns:
            Formatted string with tool results
        """
        formatted_results = "I executed the tools you requested. Here are the results:\n\n"
        
        for i, output in enumerate(tool_outputs, 1):
            tool_name = output["tool_name"]
            formatted_results += f"## Tool {i}: {tool_name}\n\n"
            
            if "error" in output:
                formatted_results += f"Error: {output['error']}\n\n"
            else:
                # Format arguments
                formatted_results += "Arguments:\n"
                for arg_name, arg_value in output["args"].items():
                    if isinstance(arg_value, str) and len(arg_value) > 100:
                        arg_snippet = arg_value[:100] + "..."
                        formatted_results += f"- {arg_name}: {arg_snippet}\n"
                    else:
                        formatted_results += f"- {arg_name}: {arg_value}\n"
                        
                # Format result based on tool type
                formatted_results += "\nResult:\n"
                result = output["result"]
                
                if tool_name == "WebSearch":
                    formatted_results += f"Found {result.get('num_results', 0)} results for query: '{output['args'].get('query', '')}'.\n\n"
                    for j, search_result in enumerate(result.get("results", []), 1):
                        formatted_results += f"{j}. **{search_result.get('title', 'Untitled')}**\n"
                        formatted_results += f"   Source: {search_result.get('source_name', search_result.get('link', 'Unknown'))}\n"
                        formatted_results += f"   Snippet: {search_result.get('snippet', 'No description')}\n\n"
                        
                elif tool_name == "InformationExtraction":
                    extracted = result.get("extracted_information", [])
                    formatted_results += f"Extraction type: {result.get('extraction_type', 'unknown')}\n"
                    formatted_results += f"Found {len(extracted)} items.\n\n"
                    
                    for item in extracted[:5]:  # Limit to prevent too long responses
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if k not in ["context"]:  # Skip verbose fields
                                    formatted_results += f"- {k}: {v}\n"
                            formatted_results += "\n"
                        else:
                            formatted_results += f"- {item}\n"
                            
                elif tool_name == "CitationGeneration":
                    if result.get("success", False):
                        formatted_results += f"Citation generated successfully.\n"
                        formatted_results += f"Citation text: {result.get('citation_text', 'Not available')}\n"
                        formatted_results += f"Style: {result.get('style', 'Unknown')}\n"
                    else:
                        formatted_results += f"Failed to generate citation: {result.get('error', 'Unknown error')}\n"
                
                else:
                    # Generic result formatting for other tools
                    if isinstance(result, dict):
                        for k, v in result.items():
                            if isinstance(v, str) and len(v) > 200:
                                v = v[:200] + "..."
                            formatted_results += f"- {k}: {v}\n"
                    else:
                        formatted_results += str(result)
                        
            formatted_results += "\n---\n\n"
            
        formatted_results += "Please continue your research using these results. If you need to use more tools, please indicate clearly which tool you want to use and how."
        
        return formatted_results
    
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
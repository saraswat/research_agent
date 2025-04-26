# tools/web_search_tool.py

"""
Tool for searching the web to gather information.
"""
from typing import Dict, List, Any, Optional, Tuple
import json
import requests
import time
import datetime
from urllib.parse import quote_plus, urlparse
import os
import re
import hashlib
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebSearchTool")

class WebSearchTool:
    """Tool for searching the web and retrieving relevant information."""
    
    def __init__(self, search_api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        """
        Initialize the WebSearchTool.
        
        Args:
            search_api_key: API key for the search service (e.g., Google Custom Search)
            search_engine_id: ID for the search engine (if using Google Custom Search)
        """
        self.search_api_key = search_api_key or os.environ.get("SEARCH_API_KEY")
        self.search_engine_id = search_engine_id or os.environ.get("SEARCH_ENGINE_ID")
        
        # Track search results to support citation generation
        self.search_history = []
        
        # Cache for search results and page content
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache retention time (15 minutes)
        self.cache_retention = 15 * 60
        
        # Supported search engines
        self.supported_engines = {
            "google": self._google_search,
            "bing": self._bing_search,
            "duckduckgo": self._duckduckgo_search,
            "mock": self._mock_search
        }
        
    def search(self, 
               query: str, 
               num_results: int = 5, 
               search_engine: str = "auto",
               cache: bool = True) -> Dict[str, Any]:
        """
        Search the web for information related to the query.
        
        Args:
            query: The search query
            num_results: Number of results to return
            search_engine: The search engine to use ('google', 'bing', 'duckduckgo', 'mock', or 'auto')
            cache: Whether to use and update the cache
            
        Returns:
            A dictionary containing search results and metadata
        """
        # Clean the query
        query = query.strip()
        
        # Check cache if enabled
        if cache:
            cached_results = self._get_from_cache(f"search_{search_engine}_{query}_{num_results}")
            if cached_results:
                logger.info(f"Using cached search results for query: {query}")
                # Track in history (even if from cache)
                self._update_search_history(query, cached_results.get("results", []))
                return cached_results
        
        # Determine which search engine to use
        if search_engine == "auto":
            if self.search_api_key and self.search_engine_id:
                search_engine = "google"
            else:
                search_engine = "mock"
        
        # Use the appropriate search function
        if search_engine in self.supported_engines:
            search_func = self.supported_engines[search_engine]
            results = search_func(query, num_results)
        else:
            # Default to mock search if unsupported engine specified
            results = self._mock_search(query, num_results)
        
        # Update cache if enabled
        if cache and results.get("success"):
            self._add_to_cache(f"search_{search_engine}_{query}_{num_results}", results)
        
        # Track search in history (if not already tracked from cache)
        if results.get("success"):
            self._update_search_history(query, results.get("results", []))
        
        return results
    
    def _update_search_history(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Add search to history for citation tracking."""
        self.search_history.append({
            "query": query,
            "time": time.time(),
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results
        })
    
    def _google_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Perform a search using Google Custom Search API."""
        if not self.search_api_key or not self.search_engine_id:
            logger.warning("Google search attempted without API key or engine ID")
            return {
                "success": False,
                "query": query,
                "error": "Google Custom Search API key and engine ID are required",
                "results": []
            }
        
        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.search_api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(num_results, 10)  # Google API limit
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            search_data = response.json()
            
            results = []
            for item in search_data.get("items", []):
                result = {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "source_name": item.get("displayLink"),
                    "published_date": None  # Not directly available from Google Search
                }
                results.append(result)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "num_results": len(results),
                "search_engine": "google"
            }
            
        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": [],
                "search_engine": "google"
            }
    
    def _bing_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Perform a search using Bing Search API."""
        bing_api_key = self.search_api_key or os.environ.get("BING_API_KEY")
        
        if not bing_api_key:
            logger.warning("Bing search attempted without API key")
            return {
                "success": False,
                "query": query,
                "error": "Bing Search API key is required",
                "results": []
            }
        
        base_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": bing_api_key
        }
        params = {
            "q": query,
            "count": min(num_results, 50)  # Bing API limit
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            search_data = response.json()
            
            results = []
            for item in search_data.get("webPages", {}).get("value", []):
                result = {
                    "title": item.get("name"),
                    "link": item.get("url"),
                    "snippet": item.get("snippet"),
                    "source_name": urlparse(item.get("url")).netloc,
                    "published_date": None  # Not available from Bing Search
                }
                results.append(result)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "num_results": len(results),
                "search_engine": "bing"
            }
            
        except Exception as e:
            logger.error(f"Bing search failed: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": [],
                "search_engine": "bing"
            }
    
    def _duckduckgo_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Perform a search using DuckDuckGo."""
        # DuckDuckGo doesn't have an official API, using a workaround with HTML parsing
        # This is not recommended for production use
        
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            for result in soup.select(".result")[:num_results]:
                title_elem = result.select_one(".result__title")
                link_elem = result.select_one(".result__url")
                snippet_elem = result.select_one(".result__snippet")
                
                title = title_elem.get_text().strip() if title_elem else "No title"
                link = link_elem.get("href") if link_elem else "#"
                snippet = snippet_elem.get_text().strip() if snippet_elem else "No description"
                
                if link.startswith("/"):
                    link = "https://duckduckgo.com" + link
                
                results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "source_name": urlparse(link).netloc,
                    "published_date": None  # Not available from DuckDuckGo
                })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "num_results": len(results),
                "search_engine": "duckduckgo"
            }
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": [],
                "search_engine": "duckduckgo"
            }
    
    def _mock_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Mock search function for demonstration purposes."""
        logger.info(f"Using mock search for query: {query}")
        
        # Generate mock results based on the query
        results = []
        for i in range(min(num_results, 5)):
            # Generate a somewhat realistic-looking mock result
            keywords = re.sub(r'[^\w\s]', '', query).split()
            title_words = []
            for _ in range(3):
                if keywords:
                    title_words.append(keywords[i % len(keywords)])
                title_words.append(["Research", "Analysis", "Guide", "Overview", "Study"][i % 5])
            
            title = " ".join(title_words).title()
            domain = ["example.com", "research.org", "wiki-knowledge.org", "info-source.net", "academic-papers.edu"][i % 5]
            
            result = {
                "title": f"{title}",
                "link": f"https://{domain}/article-{i+1}",
                "snippet": f"This is a mock snippet for {query}... Contains information about {' and '.join(keywords[:2] if len(keywords) > 1 else keywords)}. Click to learn more about this topic.",
                "source_name": domain,
                "published_date": (datetime.datetime.now() - datetime.timedelta(days=i*7)).strftime("%Y-%m-%d")
            }
            results.append(result)
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "num_results": len(results),
            "search_engine": "mock"
        }
    
    def get_page_content(self, url: str, cache: bool = True) -> Dict[str, Any]:
        """
        Retrieve and parse the content of a webpage.
        
        Args:
            url: The URL of the webpage to retrieve
            cache: Whether to use and update the cache
            
        Returns:
            A dictionary containing the webpage content and metadata
        """
        # Check cache if enabled
        if cache:
            cached_content = self._get_from_cache(f"page_{url}")
            if cached_content:
                logger.info(f"Using cached content for URL: {url}")
                return cached_content
        
        try:
            # Set headers to mimic a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Try to detect content type
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Parse HTML content
            if 'text/html' in content_type:
                result = self._parse_html_content(url, response.text)
            elif 'application/json' in content_type:
                result = self._parse_json_content(url, response.text)
            elif 'text/plain' in content_type:
                result = self._parse_text_content(url, response.text)
            else:
                # Default to HTML parsing for unknown content types
                result = self._parse_html_content(url, response.text)
            
            # Update cache if enabled
            if cache and result.get("success"):
                self._add_to_cache(f"page_{url}", result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "content": None
            }
    
    def _parse_html_content(self, url: str, html_content: str) -> Dict[str, Any]:
        """Parse HTML content and extract main content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.title.string.strip() if soup.title else "No title"
            
            # Remove script and style tags
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            
            # Extract main content
            main_content = ""
            
            # Try to find main content container
            main_elements = soup.select('main, article, #content, .content, #main, .main, .post, .article')
            
            if main_elements:
                # Use the first main element found
                main_element = main_elements[0]
                
                # Extract paragraphs from the main element
                for paragraph in main_element.find_all('p'):
                    text = paragraph.get_text().strip()
                    if text:
                        main_content += text + "\n\n"
            else:
                # Fallback to extracting all paragraphs
                for paragraph in soup.find_all('p'):
                    text = paragraph.get_text().strip()
                    if text:
                        main_content += text + "\n\n"
            
            # If still no content, get all text
            if not main_content:
                main_content = soup.get_text(separator="\n\n", strip=True)
            
            # Extract metadata
            meta_description = ""
            meta_tags = soup.find_all('meta', attrs={'name': 'description'})
            if meta_tags:
                meta_description = meta_tags[0].get('content', '')
            
            # Try to extract published date
            published_date = None
            date_metas = soup.find_all('meta', attrs={'property': ['article:published_time', 'og:published_time']})
            if date_metas:
                published_date = date_metas[0].get('content', '')
            
            # Extract domain name for source
            domain = urlparse(url).netloc
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "content": main_content,
                "source_name": domain,
                "description": meta_description,
                "published_date": published_date,
                "retrieved_date": datetime.datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            logger.error(f"Error parsing HTML content from {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": f"HTML parsing error: {str(e)}",
                "content": None
            }
    
    def _parse_json_content(self, url: str, json_content: str) -> Dict[str, Any]:
        """Parse JSON content."""
        try:
            data = json.loads(json_content)
            
            # Convert the JSON data to a formatted string for readability
            formatted_content = json.dumps(data, indent=2)
            
            return {
                "success": True,
                "url": url,
                "title": f"JSON data from {urlparse(url).netloc}",
                "content": formatted_content,
                "source_name": urlparse(url).netloc,
                "content_type": "json",
                "retrieved_date": datetime.datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            logger.error(f"Error parsing JSON content from {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": f"JSON parsing error: {str(e)}",
                "content": None
            }
    
    def _parse_text_content(self, url: str, text_content: str) -> Dict[str, Any]:
        """Parse plain text content."""
        try:
            return {
                "success": True,
                "url": url,
                "title": f"Text content from {urlparse(url).netloc}",
                "content": text_content,
                "source_name": urlparse(url).netloc,
                "content_type": "text",
                "retrieved_date": datetime.datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            logger.error(f"Error processing text content from {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": f"Text processing error: {str(e)}",
                "content": None
            }
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache if it exists and is not expired."""
        try:
            # Create a hash of the cache key for the filename
            filename = hashlib.md5(cache_key.encode()).hexdigest() + ".json"
            cache_file = os.path.join(self.cache_dir, filename)
            
            if os.path.exists(cache_file):
                # Check if cache is expired
                if time.time() - os.path.getmtime(cache_file) > self.cache_retention:
                    # Cache expired, remove it
                    os.remove(cache_file)
                    return None
                
                # Cache exists and is not expired, read it
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.warning(f"Error reading from cache: {str(e)}")
            
        return None
    
    def _add_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Add data to cache."""
        try:
            # Create a hash of the cache key for the filename
            filename = hashlib.md5(cache_key.encode()).hexdigest() + ".json"
            cache_file = os.path.join(self.cache_dir, filename)
            
            # Write data to cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")
    
    def clear_cache(self, older_than: Optional[int] = None) -> int:
        """
        Clear the search and page content cache.
        
        Args:
            older_than: Optional age in seconds; only clear items older than this
            
        Returns:
            Number of cache files removed
        """
        count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                cache_file = os.path.join(self.cache_dir, filename)
                if not older_than or (time.time() - os.path.getmtime(cache_file) > older_than):
                    os.remove(cache_file)
                    count += 1
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            
        return count
    
    def clean_expired_cache(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of expired cache files removed
        """
        return self.clear_cache(older_than=self.cache_retention)
    
    def get_citation_data(self, url: str, retries: int = 1) -> Dict[str, Any]:
        """
        Get citation data for a webpage.
        
        Args:
            url: The URL to get citation data for
            retries: Number of retries if the first attempt fails
            
        Returns:
            Dictionary with citation data
        """
        # First check if we already have this URL in search history
        for search in self.search_history:
            for result in search.get("results", []):
                if result.get("link") == url:
                    return {
                        "title": result.get("title", ""),
                        "url": url,
                        "source_name": result.get("source_name", urlparse(url).netloc),
                        "published_date": result.get("published_date"),
                        "retrieved_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "source_type": "web"
                    }
        
        # If not in history, fetch the content
        content_data = self.get_page_content(url)
        
        if content_data.get("success"):
            return {
                "title": content_data.get("title", ""),
                "url": url,
                "source_name": content_data.get("source_name", urlparse(url).netloc),
                "published_date": content_data.get("published_date"),
                "retrieved_date": content_data.get("retrieved_date", datetime.datetime.now().strftime("%Y-%m-%d")),
                "source_type": "web"
            }
        elif retries > 0:
            # Wait briefly and retry
            time.sleep(2)
            return self.get_citation_data(url, retries - 1)
        else:
            # Return basic data if all retries failed
            return {
                "title": f"Content from {urlparse(url).netloc}",
                "url": url,
                "source_name": urlparse(url).netloc,
                "retrieved_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "source_type": "web"
            }
# tools/document_analysis_tool.py

"""
Tool for processing and extracting information from various document types.
"""
from typing import Dict, Any, Optional, List, Union, BinaryIO
import os
import tempfile
import time
import datetime
import logging
import mimetypes
import hashlib
from urllib.parse import urlparse
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DocumentAnalysisTool")

class DocumentAnalysisTool:
    """Tool for analyzing and extracting text and information from documents."""
    
    def __init__(self):
        """Initialize the DocumentAnalysisTool."""
        # Keep track of analyzed documents for citation purposes
        self.analyzed_documents = []
        
        # Initialize cache directory
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "documents")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache retention time (30 minutes)
        self.cache_retention = 30 * 60
        
        # Import necessary libraries if available
        self._init_dependencies()
    
    def _init_dependencies(self):
        """Initialize document processing dependencies."""
        # PDF processing
        try:
            import PyPDF2
            self.pdf_extractor = PyPDF2.PdfReader
            logger.info("PyPDF2 loaded successfully")
        except ImportError:
            self.pdf_extractor = None
            logger.warning("PyPDF2 not available - PDF extraction will be limited")
        
        # DOCX processing
        try:
            import docx
            self.docx_extractor = docx.Document
            logger.info("python-docx loaded successfully")
        except ImportError:
            self.docx_extractor = None
            logger.warning("python-docx not available - DOCX extraction will be limited")
        
        # Web content processing
        try:
            import requests
            from bs4 import BeautifulSoup
            self.web_extractor = True
            logger.info("Web extraction dependencies loaded successfully")
        except ImportError:
            self.web_extractor = False
            logger.warning("Web extraction dependencies not available - URL extraction will be limited")
    
    def analyze(self, 
                document_source: Union[str, bytes, BinaryIO], 
                document_type: Optional[str] = None,
                cache: bool = True) -> Dict[str, Any]:
        """
        Analyze a document and extract its textual content.
        
        Args:
            document_source: File path, URL, or content of the document
            document_type: Optional type specification (pdf, docx, text, etc.)
            cache: Whether to use and update the cache
            
        Returns:
            A dictionary containing the extracted content and metadata
        """
        # Determine the document type if not provided
        if not document_type:
            document_type = self._determine_document_type(document_source)
        
        # Check cache if enabled and document_source is a string (path or URL)
        if cache and isinstance(document_source, str):
            cache_key = f"doc_{document_type}_{self._get_source_hash(document_source)}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached result for {document_source}")
                
                # Track in analysis history (even if from cache)
                if cached_result.get("success", False):
                    self._track_document(cached_result)
                    
                return cached_result
        
        # Extract content based on document type
        result = None
        if document_type == "pdf":
            result = self._extract_from_pdf(document_source)
        elif document_type == "docx":
            result = self._extract_from_docx(document_source)
        elif document_type == "text":
            result = self._extract_from_text(document_source)
        elif document_type == "url":
            result = self._extract_from_url(document_source)
        elif document_type == "html":
            result = self._extract_from_html(document_source)
        else:
            result = {
                "success": False,
                "error": f"Unsupported document type: {document_type}",
                "content": None
            }
        
        # Update cache if enabled and result was successful
        if cache and isinstance(document_source, str) and result.get("success", False):
            cache_key = f"doc_{document_type}_{self._get_source_hash(document_source)}"
            self._add_to_cache(cache_key, result)
        
        # Track document for citation purposes
        if result.get("success", False):
            self._track_document(result)
            
        return result
    
    def _track_document(self, document_data: Dict[str, Any]) -> None:
        """Add document to analysis history for citation tracking."""
        self.analyzed_documents.append({
            "source": document_data.get("document_source", "unknown"),
            "type": document_data.get("document_type", "unknown"),
            "time": time.time(),
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": {
                "title": document_data.get("title", ""),
                "author": document_data.get("author", ""),
                "date": document_data.get("date", ""),
                "pages": document_data.get("pages", 0)
            }
        })
    
    def _determine_document_type(self, document_source: Union[str, bytes, BinaryIO]) -> str:
        """Determine the document type based on the source."""
        if isinstance(document_source, str):
            # Check if it's a URL
            if document_source.startswith(('http://', 'https://')):
                return "url"
            
            # Check file extension for local files
            _, ext = os.path.splitext(document_source)
            ext = ext.lower()
            
            if ext in ['.pdf']:
                return "pdf"
            elif ext in ['.docx', '.doc']:
                return "docx"
            elif ext in ['.txt', '.md', '.csv', '.json', '.xml', '.yml', '.yaml']:
                return "text"
            elif ext in ['.html', '.htm']:
                return "html"
            else:
                # Try to determine using MIME type
                mime_type, _ = mimetypes.guess_type(document_source)
                if mime_type:
                    if mime_type == 'application/pdf':
                        return "pdf"
                    elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                                      'application/msword']:
                        return "docx"
                    elif mime_type.startswith('text/'):
                        return "text"
                    elif mime_type in ['text/html', 'application/xhtml+xml']:
                        return "html"
        
        # For bytes or file objects, try to detect content type
        if isinstance(document_source, bytes):
            # Check for PDF signature
            if document_source.startswith(b'%PDF'):
                return "pdf"
            # Check for DOCX signature (ZIP file format)
            elif document_source.startswith(b'PK\x03\x04'):
                return "docx"
            # Check for HTML signature
            elif document_source.startswith(b'<!DOCTYPE html>') or document_source.startswith(b'<html'):
                return "html"
            
        # Default to text if we can't determine
        return "text"
    
    def _extract_from_pdf(self, document_source: Union[str, bytes, BinaryIO]) -> Dict[str, Any]:
        """Extract text from a PDF document."""
        try:
            # Check if PyPDF2 is available
            if self.pdf_extractor is None:
                return self._extract_from_pdf_fallback(document_source)
            
            # Prepare document for extraction
            temp_file = None
            if isinstance(document_source, str) and os.path.exists(document_source):
                # It's a file path
                pdf_file = document_source
            else:
                # It's content or a file object, save to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                if isinstance(document_source, bytes):
                    temp_file.write(document_source)
                elif hasattr(document_source, 'read'):
                    # It's a file-like object
                    temp_file.write(document_source.read())
                else:
                    # Assume it's a string content
                    temp_file.write(document_source.encode('utf-8'))
                temp_file.close()
                pdf_file = temp_file.name
            
            # Extract text using PyPDF2
            with open(pdf_file, 'rb') as file:
                reader = self.pdf_extractor(file)
                content = ""
                metadata = reader.metadata
                pages_count = len(reader.pages)
                
                # Extract text from each page
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        content += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
            
            # Clean up temp file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            
            # Extract metadata
            title = metadata.get('/Title', os.path.basename(pdf_file) if isinstance(document_source, str) else "Untitled PDF")
            author = metadata.get('/Author', "Unknown")
            creation_date = metadata.get('/CreationDate', "")
            
            # Try to parse PDF creation date
            if creation_date and isinstance(creation_date, str):
                # PDF dates are typically in format: D:YYYYMMDDHHmmSS
                match = re.search(r'D:(\d{4})(\d{2})(\d{2})', creation_date)
                if match:
                    year, month, day = match.groups()
                    creation_date = f"{year}-{month}-{day}"
            
            result = {
                "success": True,
                "document_source": pdf_file if isinstance(document_source, str) else "pdf_content",
                "document_type": "pdf",
                "content": content,
                "title": title,
                "author": author,
                "date": creation_date,
                "pages": pages_count,
                "metadata": {
                    "title": title,
                    "author": author,
                    "creation_date": creation_date,
                    "pages": pages_count
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting from PDF: {str(e)}")
            return {
                "success": False,
                "document_source": document_source if isinstance(document_source, str) else "pdf_content",
                "document_type": "pdf",
                "error": str(e),
                "content": None
            }
    
    def _extract_from_pdf_fallback(self, document_source: Union[str, bytes, BinaryIO]) -> Dict[str, Any]:
        """Fallback method for PDF extraction when PyPDF2 is not available."""
        logger.warning("Using fallback PDF extraction - limited functionality")
        
        try:
            # Extract minimal information without PyPDF2
            if isinstance(document_source, str) and os.path.exists(document_source):
                file_path = document_source
                file_size = os.path.getsize(file_path)
                title = os.path.basename(file_path)
            else:
                file_path = "pdf_content"
                file_size = len(document_source) if isinstance(document_source, bytes) else 0
                title = "Untitled PDF"
            
            # Mock extraction with warning
            content = f"[PDF content extraction unavailable - PyPDF2 library not installed]\n\nThis appears to be a PDF document of approximately {file_size/1024:.1f} KB."
            
            result = {
                "success": True,
                "document_source": file_path,
                "document_type": "pdf",
                "content": content,
                "title": title,
                "author": "Unknown",
                "date": "",
                "pages": 0,
                "metadata": {
                    "title": title,
                    "file_size": file_size,
                    "note": "Limited extraction - PyPDF2 not available"
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in PDF fallback extraction: {str(e)}")
            return {
                "success": False,
                "document_source": document_source if isinstance(document_source, str) else "pdf_content",
                "document_type": "pdf",
                "error": f"PDF extraction failed (PyPDF2 not available): {str(e)}",
                "content": None
            }
    
    def _extract_from_docx(self, document_source: Union[str, bytes, BinaryIO]) -> Dict[str, Any]:
        """Extract text from a DOCX document."""
        try:
            # Check if python-docx is available
            if self.docx_extractor is None:
                return self._extract_from_docx_fallback(document_source)
            
            # Prepare document for extraction
            temp_file = None
            if isinstance(document_source, str) and os.path.exists(document_source):
                # It's a file path
                docx_file = document_source
            else:
                # It's content or a file object, save to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
                if isinstance(document_source, bytes):
                    temp_file.write(document_source)
                elif hasattr(document_source, 'read'):
                    # It's a file-like object
                    temp_file.write(document_source.read())
                else:
                    # Assume it's a string content
                    temp_file.write(document_source.encode('utf-8'))
                temp_file.close()
                docx_file = temp_file.name
            
            # Extract text using python-docx
            doc = self.docx_extractor(docx_file)
            content = ""
            
            # Extract document properties
            core_properties = doc.core_properties
            title = core_properties.title or os.path.basename(docx_file) if isinstance(document_source, str) else "Untitled Document"
            author = core_properties.author or "Unknown"
            created = core_properties.created
            
            # Format creation date if available
            created_date = ""
            if created:
                created_date = created.strftime("%Y-%m-%d")
            
            # Extract paragraphs from the document
            for para in doc.paragraphs:
                if para.text.strip():
                    content += para.text + "\n\n"
            
            # Extract tables from the document
            for table in doc.tables:
                content += "--- Table ---\n"
                for row in table.rows:
                    row_texts = [cell.text for cell in row.cells]
                    content += " | ".join(row_texts) + "\n"
                content += "\n"
            
            # Clean up temp file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            
            result = {
                "success": True,
                "document_source": docx_file if isinstance(document_source, str) else "docx_content",
                "document_type": "docx",
                "content": content,
                "title": title,
                "author": author,
                "date": created_date,
                "metadata": {
                    "title": title,
                    "author": author,
                    "creation_date": created_date
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting from DOCX: {str(e)}")
            return {
                "success": False,
                "document_source": document_source if isinstance(document_source, str) else "docx_content",
                "document_type": "docx",
                "error": str(e),
                "content": None
            }
    
    def _extract_from_docx_fallback(self, document_source: Union[str, bytes, BinaryIO]) -> Dict[str, Any]:
        """Fallback method for DOCX extraction when python-docx is not available."""
        logger.warning("Using fallback DOCX extraction - limited functionality")
        
        try:
            # Extract minimal information without python-docx
            if isinstance(document_source, str) and os.path.exists(document_source):
                file_path = document_source
                file_size = os.path.getsize(file_path)
                title = os.path.basename(file_path)
            else:
                file_path = "docx_content"
                file_size = len(document_source) if isinstance(document_source, bytes) else 0
                title = "Untitled Document"
            
            # Mock extraction with warning
            content = f"[DOCX content extraction unavailable - python-docx library not installed]\n\nThis appears to be a Word document of approximately {file_size/1024:.1f} KB."
            
            result = {
                "success": True,
                "document_source": file_path,
                "document_type": "docx",
                "content": content,
                "title": title,
                "author": "Unknown",
                "date": "",
                "metadata": {
                    "title": title,
                    "file_size": file_size,
                    "note": "Limited extraction - python-docx not available"
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in DOCX fallback extraction: {str(e)}")
            return {
                "success": False,
                "document_source": document_source if isinstance(document_source, str) else "docx_content",
                "document_type": "docx",
                "error": f"DOCX extraction failed (python-docx not available): {str(e)}",
                "content": None
            }
    
    def _extract_from_text(self, document_source: Union[str, bytes, BinaryIO]) -> Dict[str, Any]:
        """Extract content from a text document or string."""
        try:
            content = ""
            source_type = ""
            
            # Handle different input types
            if isinstance(document_source, str):
                if os.path.exists(document_source):
                    # It's a file path
                    with open(document_source, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    source_type = "file"
                    title = os.path.basename(document_source)
                else:
                    # It's raw text
                    content = document_source
                    source_type = "raw_text"
                    title = "Text Content"
            elif isinstance(document_source, bytes):
                # It's bytes content
                content = document_source.decode('utf-8', errors='replace')
                source_type = "bytes"
                title = "Text Content"
            elif hasattr(document_source, 'read'):
                # It's a file-like object
                content = document_source.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                source_type = "file_object"
                title = "Text Content"
            
            # Try to determine if it's a specific format
            format_type = "plain_text"
            if content.strip().startswith("{") and content.strip().endswith("}"):
                try:
                    # Try to parse as JSON
                    json.loads(content)
                    format_type = "json"
                except json.JSONDecodeError:
                    pass
            elif content.strip().startswith("<") and content.strip().endswith(">"):
                # Might be XML or HTML
                if "<html" in content.lower() or "<!doctype html" in content.lower():
                    format_type = "html"
                else:
                    format_type = "xml"
            elif "---" in content[:50] and ":" in content[:100]:
                # Might be YAML/Markdown with frontmatter
                format_type = "yaml_markdown"
            
            # Additional metadata for specific formats
            metadata = {
                "source_type": source_type,
                "format_type": format_type,
                "length": len(content),
                "line_count": content.count('\n') + 1
            }
            
            result = {
                "success": True,
                "document_source": document_source if isinstance(document_source, str) and source_type == "file" else "text_content",
                "document_type": "text",
                "content": content,
                "title": title,
                "format": format_type,
                "metadata": metadata
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting from text: {str(e)}")
            return {
                "success": False,
                "document_source": document_source if isinstance(document_source, str) else "text_content",
                "document_type": "text",
                "error": str(e),
                "content": None
            }
    
    def _extract_from_url(self, document_source: str) -> Dict[str, Any]:
        """Extract content from a URL (web page)."""
        try:
            # Check if web extraction dependencies are available
            if not self.web_extractor:
                return {
                    "success": False,
                    "document_source": document_source,
                    "document_type": "url",
                    "error": "Web extraction dependencies (requests, BeautifulSoup) not available",
                    "content": None
                }
            
            import requests
            from bs4 import BeautifulSoup
            
            # Set headers to mimic a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Fetch the webpage
            response = requests.get(document_source, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Detect content type
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Handle different content types
            if 'application/pdf' in content_type:
                # It's a PDF file
                return self._extract_from_pdf(response.content)
            elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                # It's a DOCX file
                return self._extract_from_docx(response.content)
            elif 'text/html' in content_type:
                # It's an HTML page
                return self._extract_from_html(response.text, document_source)
            elif 'application/json' in content_type:
                # It's JSON data
                return self._extract_from_text(response.text)
            elif 'text/' in content_type:
                # It's some kind of text
                return self._extract_from_text(response.text)
            else:
                # Unknown content type, try HTML parsing as default
                return self._extract_from_html(response.text, document_source)
                
        except Exception as e:
            logger.error(f"Error extracting from URL {document_source}: {str(e)}")
            return {
                "success": False,
                "document_source": document_source,
                "document_type": "url",
                "error": str(e),
                "content": None
            }
    
    def _extract_from_html(self, document_source: Union[str, bytes], url: Optional[str] = None) -> Dict[str, Any]:
        """Extract content from an HTML document."""
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return {
                    "success": False,
                    "document_source": url or "html_content",
                    "document_type": "html",
                    "error": "BeautifulSoup library not available for HTML parsing",
                    "content": None
                }
            
            # Parse HTML content
            if isinstance(document_source, bytes):
                html_content = document_source.decode('utf-8', errors='replace')
            else:
                html_content = document_source
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string.strip()
            elif url:
                title = f"Content from {urlparse(url).netloc}"
            else:
                title = "HTML Content"
            
            # Remove script, style, and other non-content tags
            for tag in soup(['script', 'style', 'meta', 'link', 'noscript', 'iframe']):
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
                
                # Extract headers
                for header in main_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    text = header.get_text().strip()
                    if text:
                        main_content += f"{'#' * int(header.name[1])} {text}\n\n"
                
                # Extract lists
                for list_elem in main_element.find_all(['ul', 'ol']):
                    for item in list_elem.find_all('li'):
                        text = item.get_text().strip()
                        if text:
                            main_content += f"- {text}\n"
                    main_content += "\n"
            else:
                # Fallback to extracting all paragraphs
                for paragraph in soup.find_all('p'):
                    text = paragraph.get_text().strip()
                    if text:
                        main_content += text + "\n\n"
                
                # Extract headers
                for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    text = header.get_text().strip()
                    if text:
                        main_content += f"{'#' * int(header.name[1])} {text}\n\n"
            
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
            
            # Try to extract author
            author = "Unknown"
            author_metas = soup.find_all('meta', attrs={'name': 'author'})
            if author_metas:
                author = author_metas[0].get('content', 'Unknown')
            
            # Extract domain name for source
            domain = urlparse(url).netloc if url else "html_document"
            
            result = {
                "success": True,
                "document_source": url or "html_content",
                "document_type": "html",
                "content": main_content,
                "title": title,
                "author": author,
                "date": published_date,
                "source_name": domain,
                "description": meta_description,
                "metadata": {
                    "title": title,
                    "author": author,
                    "publication_date": published_date,
                    "description": meta_description,
                    "source": domain
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting from HTML: {str(e)}")
            return {
                "success": False,
                "document_source": url or "html_content",
                "document_type": "html",
                "error": str(e),
                "content": None
            }
    
    def _get_source_hash(self, source: str) -> str:
        """Generate a hash for document source to use as cache key."""
        return hashlib.md5(source.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get document data from cache if it exists."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
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
        """Add document data to cache."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            # Write data to cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")
    
    def clear_cache(self, older_than: Optional[int] = None) -> int:
        """
        Clear the document cache.
        
        Args:
            older_than: Optional age in seconds; only clear items older than this
            
        Returns:
            Number of cache files removed
        """
        count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    cache_file = os.path.join(self.cache_dir, filename)
                    if not older_than or (time.time() - os.path.getmtime(cache_file) > older_than):
                        os.remove(cache_file)
                        count += 1
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            
        return count
    
    def get_document_metadata(self, document_source: Union[str, bytes, BinaryIO], 
                             document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a document without extracting full content.
        
        Args:
            document_source: File path, URL, or content of the document
            document_type: Optional type specification (pdf, docx, text, etc.)
            
        Returns:
            Dictionary with document metadata
        """
        # First check if we already have this document in analysis history
        if isinstance(document_source, str):
            for doc in self.analyzed_documents:
                if doc["source"] == document_source:
                    return {
                        "document_type": doc["type"],
                        "title": doc["metadata"].get("title", ""),
                        "author": doc["metadata"].get("author", ""),
                        "date": doc["metadata"].get("date", ""),
                        "pages": doc["metadata"].get("pages", 0),
                        "source": document_source
                    }
        
        # Analyze the document but extract only metadata
        result = self.analyze(document_source, document_type)
        
        if result.get("success"):
            return {
                "document_type": result.get("document_type", "unknown"),
                "title": result.get("title", ""),
                "author": result.get("author", ""),
                "date": result.get("date", ""),
                "pages": result.get("pages", 0),
                "source": result.get("document_source", "")
            }
        else:
            return {
                "document_type": "unknown",
                "title": "Unknown document",
                "author": "Unknown",
                "date": "",
                "pages": 0,
                "source": document_source if isinstance(document_source, str) else "document_content",
                "error": result.get("error", "Unknown error")
            }
# tools/report_generation_tool.py
"""
Tool for formatting and structuring the final research report.
"""
from typing import Dict, List, Any, Optional
import io
import os
import json
import re
from datetime import datetime

class ReportGenerationTool:
    """Tool for generating structured research reports with citations."""
    
    def __init__(self):
        """Initialize the ReportGenerationTool."""
        self.supported_formats = ["markdown", "html", "text", "pdf", "latex"]
        self.report_templates = {
            "academic": self._academic_template,
            "business": self._business_template,
            "technical": self._technical_template,
            "default": self._default_template
        }
        self.section_order = [
            "abstract", 
            "introduction", 
            "literature_review",
            "methodology", 
            "results", 
            "discussion", 
            "conclusion", 
            "references"
        ]
        # Track generated reports
        self.generated_reports = []
    
    def generate_report(self, 
                        title: str, 
                        sections: List[Dict[str, Any]], 
                        citations: List[Dict[str, Any]], 
                        format: str = "markdown",
                        template: str = "default",
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a structured research report with proper citations.
        
        Args:
            title: The title of the report
            sections: List of section dictionaries with content and claims
            citations: List of citation dictionaries
            format: Output format (markdown, html, txt, pdf, latex)
            template: Report template to use (academic, business, technical, default)
            metadata: Optional metadata for the report (author, date, etc.)
            
        Returns:
            A dictionary containing the generated report
        """
        # Validate and normalize format
        format = format.lower()
        if format not in self.supported_formats:
            format = "markdown"  # Default to markdown if unsupported format
            
        # Validate and normalize template
        template = template.lower()
        if template not in self.report_templates:
            template = "default"  # Default template if unsupported
            
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Add default metadata if not provided
        if "date" not in metadata:
            metadata["date"] = datetime.now().strftime("%Y-%m-%d")
        if "author" not in metadata:
            metadata["author"] = "Research Agent"
            
        # Sort sections according to standard academic order if type is provided
        if any("type" in section for section in sections):
            sections = sorted(
                sections, 
                key=lambda s: self.section_order.index(s.get("type", "other")) 
                if s.get("type") in self.section_order else 999
            )
            
        # Generate report using appropriate format
        if format == "markdown":
            report_content = self._generate_markdown_report(title, sections, citations, metadata, template)
        elif format == "html":
            report_content = self._generate_html_report(title, sections, citations, metadata, template)
        elif format == "latex":
            report_content = self._generate_latex_report(title, sections, citations, metadata, template)
        elif format == "pdf":
            # For PDF, we generate LaTeX first, then convert to PDF if possible
            latex_content = self._generate_latex_report(title, sections, citations, metadata, template)
            report_content = self._convert_latex_to_pdf(latex_content, title)
            if isinstance(report_content, dict) and "error" in report_content:
                # If PDF conversion failed, fall back to LaTeX
                report_content = latex_content
                format = "latex"  # Update format to match actual content
        else:  # default to plain text
            report_content = self._generate_text_report(title, sections, citations, metadata, template)
            
        # Store report in history
        self.generated_reports.append({
            "title": title,
            "format": format,
            "template": template,
            "date": metadata.get("date"),
            "sections_count": len(sections),
            "citations_count": len(citations),
            "timestamp": datetime.now().isoformat()
        })
        
        # Return report data
        return {
            "title": title,
            "format": format,
            "content": report_content,
            "metadata": metadata,
            "template": template,
            "sections_count": len(sections),
            "citations_count": len(citations),
            "word_count": self._count_words(report_content) if isinstance(report_content, str) else None
        }
    
    def _count_words(self, text: str) -> int:
        """Count the number of words in the text."""
        # Remove special characters and split by whitespace
        return len(re.findall(r'\w+', text))
    
    def _convert_latex_to_pdf(self, latex_content: str, title: str) -> str:
        """Convert LaTeX content to PDF format."""
        try:
            import subprocess
            
            # Create temporary directory for LaTeX compilation
            temp_dir = os.path.join(os.getcwd(), "temp_latex")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create LaTeX file
            tex_file = os.path.join(temp_dir, f"{title.replace(' ', '_')}.tex")
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(latex_content)
            
            # Run pdflatex to generate PDF
            process = subprocess.Popen(
                ["pdflatex", "-interaction=nonstopmode", tex_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_dir
            )
            stdout, stderr = process.communicate()
            
            # Check if PDF was generated
            pdf_file = os.path.join(temp_dir, f"{title.replace(' ', '_')}.pdf")
            if os.path.exists(pdf_file):
                # Read PDF binary content
                with open(pdf_file, "rb") as f:
                    pdf_content = f.read()
                
                # Clean up temporary files
                for f in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except:
                        pass
                os.rmdir(temp_dir)
                
                return pdf_content
            else:
                # If PDF generation failed, return error
                return {
                    "error": "PDF generation failed",
                    "details": stdout.decode("utf-8") + stderr.decode("utf-8")
                }
                
        except ImportError:
            return {"error": "pdflatex is not installed"}
        except Exception as e:
            return {"error": f"PDF generation failed: {str(e)}"}
    
    def _generate_markdown_report(self, 
                                  title: str, 
                                  sections: List[Dict[str, Any]], 
                                  citations: List[Dict[str, Any]],
                                  metadata: Dict[str, Any],
                                  template: str) -> str:
        """Generate a research report in Markdown format using the specified template."""
        # Call the appropriate template function
        template_func = self.report_templates.get(template, self._default_template)
        return template_func(title, sections, citations, metadata, "markdown")
    
    def _generate_html_report(self, 
                             title: str, 
                             sections: List[Dict[str, Any]], 
                             citations: List[Dict[str, Any]],
                             metadata: Dict[str, Any],
                             template: str) -> str:
        """Generate a research report in HTML format using the specified template."""
        # Call the appropriate template function
        template_func = self.report_templates.get(template, self._default_template)
        return template_func(title, sections, citations, metadata, "html")
    
    def _generate_text_report(self, 
                             title: str, 
                             sections: List[Dict[str, Any]], 
                             citations: List[Dict[str, Any]],
                             metadata: Dict[str, Any],
                             template: str) -> str:
        """Generate a research report in plain text format using the specified template."""
        # Call the appropriate template function
        template_func = self.report_templates.get(template, self._default_template)
        return template_func(title, sections, citations, metadata, "text")
    
    def _generate_latex_report(self, 
                              title: str, 
                              sections: List[Dict[str, Any]], 
                              citations: List[Dict[str, Any]],
                              metadata: Dict[str, Any],
                              template: str) -> str:
        """Generate a research report in LaTeX format using the specified template."""
        # Call the appropriate template function
        template_func = self.report_templates.get(template, self._default_template)
        return template_func(title, sections, citations, metadata, "latex")
    
    def _default_template(self, 
                         title: str, 
                         sections: List[Dict[str, Any]], 
                         citations: List[Dict[str, Any]],
                         metadata: Dict[str, Any],
                         format: str) -> str:
        """Default template for research reports."""
        if format == "markdown":
            return self._default_markdown_template(title, sections, citations, metadata)
        elif format == "html":
            return self._default_html_template(title, sections, citations, metadata)
        elif format == "latex":
            return self._default_latex_template(title, sections, citations, metadata)
        else:  # default to plain text
            return self._default_text_template(title, sections, citations, metadata)
    
    def _academic_template(self, 
                         title: str, 
                         sections: List[Dict[str, Any]], 
                         citations: List[Dict[str, Any]],
                         metadata: Dict[str, Any],
                         format: str) -> str:
        """Academic template for research reports."""
        if format == "markdown":
            return self._academic_markdown_template(title, sections, citations, metadata)
        elif format == "html":
            return self._academic_html_template(title, sections, citations, metadata)
        elif format == "latex":
            return self._academic_latex_template(title, sections, citations, metadata)
        else:  # default to plain text
            return self._academic_text_template(title, sections, citations, metadata)
    
    def _business_template(self, 
                         title: str, 
                         sections: List[Dict[str, Any]], 
                         citations: List[Dict[str, Any]],
                         metadata: Dict[str, Any],
                         format: str) -> str:
        """Business template for research reports."""
        if format == "markdown":
            return self._business_markdown_template(title, sections, citations, metadata)
        elif format == "html":
            return self._business_html_template(title, sections, citations, metadata)
        elif format == "latex":
            return self._business_latex_template(title, sections, citations, metadata)
        else:  # default to plain text
            return self._business_text_template(title, sections, citations, metadata)
    
    def _technical_template(self, 
                          title: str, 
                          sections: List[Dict[str, Any]], 
                          citations: List[Dict[str, Any]],
                          metadata: Dict[str, Any],
                          format: str) -> str:
        """Technical template for research reports."""
        if format == "markdown":
            return self._technical_markdown_template(title, sections, citations, metadata)
        elif format == "html":
            return self._technical_html_template(title, sections, citations, metadata)
        elif format == "latex":
            return self._technical_latex_template(title, sections, citations, metadata)
        else:  # default to plain text
            return self._technical_text_template(title, sections, citations, metadata)
    
    def _default_markdown_template(self, 
                                  title: str, 
                                  sections: List[Dict[str, Any]], 
                                  citations: List[Dict[str, Any]],
                                  metadata: Dict[str, Any]) -> str:
        """Generate a research report in Markdown format using the default template."""
        report = f"# {title}\n\n"
        
        # Add metadata
        if metadata:
            report += f"**Author:** {metadata.get('author', 'Research Agent')}\n"
            report += f"**Date:** {metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}\n\n"
        
        # Add abstract if present
        abstract_section = next((s for s in sections if s.get("type") == "abstract"), None)
        if abstract_section:
            report += f"## Abstract\n\n{abstract_section['content']}\n\n"
        
        # Add introduction if present
        intro_section = next((s for s in sections if s.get("type") == "introduction"), None)
        if intro_section:
            report += f"## Introduction\n\n{intro_section['content']}\n\n"
        
        # Add main content sections
        main_sections = [s for s in sections if s.get("type") not in ["abstract", "introduction", "conclusion", "references"]]
        for section in main_sections:
            report += f"## {section['title']}\n\n"
            
            # Add content with citations
            content = section['content']
            
            # If there are claims associated with this section, add citations
            if 'claims' in section:
                for claim in section['claims']:
                    if 'citation_id' in claim and claim['citation_id'] < len(citations):
                        citation = citations[claim['citation_id']]
                        # Replace the claim text with the claim + citation
                        content = content.replace(
                            claim['text'], 
                            f"{claim['text']} {citation['in_text_citation']}"
                        )
            
            report += f"{content}\n\n"
            
            # Add subsections if present
            if 'subsections' in section:
                for subsection in section['subsections']:
                    report += f"### {subsection['title']}\n\n{subsection['content']}\n\n"
        
        # Add conclusion if present
        conclusion_section = next((s for s in sections if s.get("type") == "conclusion"), None)
        if conclusion_section:
            report += f"## Conclusion\n\n{conclusion_section['content']}\n\n"
        
        # Add references
        report += "## References\n\n"
        
        # Collect all unique references
        unique_references = {}
        for citation in citations:
            if 'reference' in citation:
                unique_references[citation['reference']] = True
        
        # Add references in alphabetical order
        for reference in sorted(unique_references.keys()):
            report += f"{reference}\n\n"
        
        return report
    
    def _default_html_template(self, 
                              title: str, 
                              sections: List[Dict[str, Any]], 
                              citations: List[Dict[str, Any]],
                              metadata: Dict[str, Any]) -> str:
        """Generate a research report in HTML format using the default template."""
        report = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 2em; }}
        h1 {{ color: #333; text-align: center; }}
        h2 {{ color: #444; margin-top: 1.5em; }}
        h3 {{ color: #555; }}
        .abstract {{ font-style: italic; margin-bottom: 2em; }}
        .citation {{ vertical-align: super; font-size: 0.8em; }}
        .references {{ margin-top: 2em; border-top: 1px solid #ccc; padding-top: 1em; }}
        .reference {{ margin-bottom: 0.5em; }}
        .metadata {{ text-align: center; margin-bottom: 2em; color: #666; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="metadata">
        <p>Author: {metadata.get('author', 'Research Agent')}<br>
        Date: {metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}</p>
    </div>
"""
        
        # Add abstract if present
        abstract_section = next((s for s in sections if s.get("type") == "abstract"), None)
        if abstract_section:
            report += f"""    <div class="abstract">
        <h2>Abstract</h2>
        <p>{abstract_section['content']}</p>
    </div>
"""
        
        # Add introduction if present
        intro_section = next((s for s in sections if s.get("type") == "introduction"), None)
        if intro_section:
            report += f"    <h2>Introduction</h2>\n    <p>{intro_section['content']}</p>\n"
        
        # Add main content sections
        main_sections = [s for s in sections if s.get("type") not in ["abstract", "introduction", "conclusion", "references"]]
        for section in main_sections:
            report += f"    <h2>{section['title']}</h2>\n"
            
            # Add content with citations
            content = section['content']
            
            # If there are claims associated with this section, add citations
            if 'claims' in section:
                for i, claim in enumerate(section['claims']):
                    if 'citation_id' in claim and claim['citation_id'] < len(citations):
                        citation = citations[claim['citation_id']]
                        # Replace the claim text with the claim + citation
                        content = content.replace(
                            claim['text'], 
                            f"{claim['text']} <sup class=\"citation\">[{i+1}]</sup>"
                        )
            
            report += f"    <p>{content}</p>\n"
            
            # Add subsections if present
            if 'subsections' in section:
                for subsection in section['subsections']:
                    report += f"    <h3>{subsection['title']}</h3>\n    <p>{subsection['content']}</p>\n"
        
        # Add conclusion if present
        conclusion_section = next((s for s in sections if s.get("type") == "conclusion"), None)
        if conclusion_section:
            report += f"    <h2>Conclusion</h2>\n    <p>{conclusion_section['content']}</p>\n"
        
        # Add references
        report += """    <div class="references">
        <h2>References</h2>
"""
        
        # Collect all unique references
        unique_references = {}
        for i, citation in enumerate(citations):
            if 'reference' in citation:
                unique_references[i] = citation['reference']
        
        # Add references with numbering
        for i, reference in sorted(unique_references.items()):
            report += f"        <p class=\"reference\">[{i+1}] {reference}</p>\n"
        
        report += "    </div>\n</body>\n</html>"
        
        return report
    
    def _default_text_template(self, 
                              title: str, 
                              sections: List[Dict[str, Any]], 
                              citations: List[Dict[str, Any]],
                              metadata: Dict[str, Any]) -> str:
        """Generate a research report in plain text format using the default template."""
        report = f"{title.upper()}\n{'=' * len(title)}\n\n"
        
        # Add metadata
        report += f"Author: {metadata.get('author', 'Research Agent')}\n"
        report += f"Date: {metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}\n\n"
        
        # Add abstract if present
        abstract_section = next((s for s in sections if s.get("type") == "abstract"), None)
        if abstract_section:
            report += f"ABSTRACT\n--------\n\n{abstract_section['content']}\n\n"
        
        # Add introduction if present
        intro_section = next((s for s in sections if s.get("type") == "introduction"), None)
        if intro_section:
            report += f"INTRODUCTION\n------------\n\n{intro_section['content']}\n\n"
        
        # Add main content sections
        main_sections = [s for s in sections if s.get("type") not in ["abstract", "introduction", "conclusion", "references"]]
        for section in main_sections:
            report += f"{section['title'].upper()}\n{'-' * len(section['title'])}\n\n"
            
            # Add content with citations
            content = section['content']
            
            # If there are claims associated with this section, add citations
            if 'claims' in section:
                for claim in section['claims']:
                    if 'citation_id' in claim and claim['citation_id'] < len(citations):
                        citation = citations[claim['citation_id']]
                        # Replace the claim text with the claim + citation
                        content = content.replace(
                            claim['text'], 
                            f"{claim['text']} {citation['in_text_citation']}"
                        )
            
            report += f"{content}\n\n"
            
            # Add subsections if present
            if 'subsections' in section:
                for subsection in section['subsections']:
                    report += f"{subsection['title']}\n{'^' * len(subsection['title'])}\n\n{subsection['content']}\n\n"
        
        # Add conclusion if present
        conclusion_section = next((s for s in sections if s.get("type") == "conclusion"), None)
        if conclusion_section:
            report += f"CONCLUSION\n----------\n\n{conclusion_section['content']}\n\n"
        
        # Add references
        report += "REFERENCES\n----------\n\n"
        
        # Collect all unique references
        unique_references = {}
        for citation in citations:
            if 'reference' in citation:
                unique_references[citation['reference']] = True
        
        # Add references in alphabetical order
        for reference in sorted(unique_references.keys()):
            report += f"{reference}\n\n"
        
        return report
    
    def _default_latex_template(self, 
                               title: str, 
                               sections: List[Dict[str, Any]], 
                               citations: List[Dict[str, Any]],
                               metadata: Dict[str, Any]) -> str:
        """Generate a research report in LaTeX format using the default template."""
        report = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{graphicx}

\geometry{a4paper, margin=1in}
\doublespacing

\title{""" + title + r"""}
\author{""" + metadata.get('author', 'Research Agent') + r"""}
\date{""" + metadata.get('date', datetime.now().strftime('%Y-%m-%d')) + r"""}

\begin{document}

\maketitle

"""
        
        # Add abstract if present
        abstract_section = next((s for s in sections if s.get("type") == "abstract"), None)
        if abstract_section:
            report += r"\begin{abstract}" + "\n" + abstract_section['content'] + "\n" + r"\end{abstract}" + "\n\n"
        
        # Add introduction if present
        intro_section = next((s for s in sections if s.get("type") == "introduction"), None)
        if intro_section:
            report += r"\section{Introduction}" + "\n\n" + intro_section['content'] + "\n\n"
        
        # Add main content sections
        main_sections = [s for s in sections if s.get("type") not in ["abstract", "introduction", "conclusion", "references"]]
        for section in main_sections:
            report += r"\section{" + section['title'] + "}\n\n"
            
            # Add content with citations
            content = section['content']
            
            # If there are claims associated with this section, add citations
            if 'claims' in section:
                for claim in section['claims']:
                    if 'citation_id' in claim and claim['citation_id'] < len(citations):
                        citation = citations[claim['citation_id']]
                        # Replace the claim text with the claim + citation (LaTeX format)
                        latex_citation = r"\cite{ref" + str(claim['citation_id']) + "}"
                        content = content.replace(
                            claim['text'], 
                            f"{claim['text']} {latex_citation}"
                        )
            
            report += f"{content}\n\n"
            
            # Add subsections if present
            if 'subsections' in section:
                for subsection in section['subsections']:
                    report += r"\subsection{" + subsection['title'] + "}\n\n" + subsection['content'] + "\n\n"
        
        # Add conclusion if present
        conclusion_section = next((s for s in sections if s.get("type") == "conclusion"), None)
        if conclusion_section:
            report += r"\section{Conclusion}" + "\n\n" + conclusion_section['content'] + "\n\n"
        
        # Add references
        report += r"\begin{thebibliography}{99}" + "\n\n"
        
        # Add bibliography entries
        for i, citation in enumerate(citations):
            if 'reference' in citation:
                # Format as BibTeX-style entry
                report += r"\bibitem{ref" + str(i) + "} " + citation['reference'] + "\n\n"
        
        report += r"\end{thebibliography}" + "\n\n"
        report += r"\end{document}"
        
        return report
    
    # Academic template implementations
    def _academic_markdown_template(self, 
                                  title: str, 
                                  sections: List[Dict[str, Any]], 
                                  citations: List[Dict[str, Any]],
                                  metadata: Dict[str, Any]) -> str:
        """Generate an academic research report in Markdown format."""
        # Academic reports typically include:
        # - Title page with author affiliations
        # - Abstract
        # - Keywords
        # - Introduction
        # - Literature Review
        # - Methodology
        # - Results
        # - Discussion
        # - Conclusion
        # - References
        
        report = f"# {title}\n\n"
        
        # Add metadata
        report += f"**Author:** {metadata.get('author', 'Research Agent')}\n"
        if 'affiliation' in metadata:
            report += f"**Affiliation:** {metadata.get('affiliation')}\n"
        report += f"**Date:** {metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}\n\n"
        
        # Add abstract if present
        abstract_section = next((s for s in sections if s.get("type") == "abstract"), None)
        if abstract_section:
            report += f"## Abstract\n\n{abstract_section['content']}\n\n"
            
        # Add keywords if present in metadata
        if 'keywords' in metadata:
            keywords = metadata['keywords']
            if isinstance(keywords, list):
                keywords = ', '.join(keywords)
            report += f"**Keywords:** {keywords}\n\n"
        
        # Add introduction
        intro_section = next((s for s in sections if s.get("type") == "introduction"), None)
        if intro_section:
            report += f"## Introduction\n\n{intro_section['content']}\n\n"
        
        # Add literature review if present
        lit_review_section = next((s for s in sections if s.get("type") == "literature_review"), None)
        if lit_review_section:
            report += f"## Literature Review\n\n{lit_review_section['content']}\n\n"
        
        # Add methodology if present
        method_section = next((s for s in sections if s.get("type") == "methodology"), None)
        if method_section:
            report += f"## Methodology\n\n{method_section['content']}\n\n"
        
        # Add results if present
        results_section = next((s for s in sections if s.get("type") == "results"), None)
        if results_section:
            report += f"## Results\n\n{results_section['content']}\n\n"
        
        # Add discussion if present
        discussion_section = next((s for s in sections if s.get("type") == "discussion"), None)
        if discussion_section:
            report += f"## Discussion\n\n{discussion_section['content']}\n\n"
        
        # Add other content sections
        other_sections = [s for s in sections if s.get("type") not in [
            "abstract", "introduction", "literature_review", "methodology", 
            "results", "discussion", "conclusion", "references"
        ]]
        
        for section in other_sections:
            report += f"## {section['title']}\n\n"
            
            # Add content with citations
            content = section['content']
            
            # Process citations
            if 'claims' in section:
                for claim in section['claims']:
                    if 'citation_id' in claim and claim['citation_id'] < len(citations):
                        citation = citations[claim['citation_id']]
                        content = content.replace(
                            claim['text'], 
                            f"{claim['text']} {citation['in_text_citation']}"
                        )
            
            report += f"{content}\n\n"
            
            # Add subsections
            if 'subsections' in section:
                for subsection in section['subsections']:
                    report += f"### {subsection['title']}\n\n{subsection['content']}\n\n"
        
        # Add conclusion
        conclusion_section = next((s for s in sections if s.get("type") == "conclusion"), None)
        if conclusion_section:
            report += f"## Conclusion\n\n{conclusion_section['content']}\n\n"
        
        # Add references
        report += "## References\n\n"
        
        # Collect and sort references
        unique_references = {}
        for citation in citations:
            if 'reference' in citation:
                unique_references[citation['reference']] = True
        
        for reference in sorted(unique_references.keys()):
            report += f"{reference}\n\n"
        
        # Add appendices if present in metadata
        if 'appendices' in metadata:
            report += "## Appendices\n\n"
            appendices = metadata['appendices']
            if isinstance(appendices, list):
                for i, appendix in enumerate(appendices):
                    if isinstance(appendix, dict) and 'title' in appendix and 'content' in appendix:
                        report += f"### Appendix {chr(65+i)}: {appendix['title']}\n\n{appendix['content']}\n\n"
        
        return report
    
    # Business template implementations
    def _business_markdown_template(self, 
                                   title: str, 
                                   sections: List[Dict[str, Any]], 
                                   citations: List[Dict[str, Any]],
                                   metadata: Dict[str, Any]) -> str:
        """Generate a business research report in Markdown format."""
        # Business reports typically include:
        # - Executive Summary
        # - Introduction/Background
        # - Key Findings
        # - Market Analysis
        # - Recommendations
        # - Implementation Plan
        # - Conclusion
        # - References
        
        report = f"# {title}\n\n"
        
        # Add metadata
        report += f"**Prepared by:** {metadata.get('author', 'Research Agent')}\n"
        if 'company' in metadata:
            report += f"**Company:** {metadata.get('company')}\n"
        report += f"**Date:** {metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}\n\n"
        
        # Table of Contents
        report += "## Table of Contents\n\n"
        
        toc_items = ["Executive Summary", "Introduction", "Key Findings"]
        
        # Add custom sections to TOC
        for section in sections:
            if section.get("type") not in ["abstract", "executive_summary", "introduction"]:
                toc_items.append(section.get("title", "Section"))
        
        # Add final TOC items
        toc_items.extend(["Recommendations", "Conclusion", "References"])
        
        # Generate TOC
        for i, item in enumerate(toc_items):
            report += f"{i+1}. {item}\n"
        
        report += "\n"
        
        # Add executive summary (use abstract if present, otherwise look for executive_summary)
        exec_summary = next((s for s in sections if s.get("type") == "executive_summary"), 
                           next((s for s in sections if s.get("type") == "abstract"), None))
        
        if exec_summary:
            report += f"## Executive Summary\n\n{exec_summary['content']}\n\n"
        
        # Add introduction
        intro_section = next((s for s in sections if s.get("type") == "introduction"), None)
        if intro_section:
            report += f"## Introduction\n\n{intro_section['content']}\n\n"
        
        # Add key findings section
        findings_section = next((s for s in sections if s.get("type") == "key_findings"), None)
        if findings_section:
            report += f"## Key Findings\n\n{findings_section['content']}\n\n"
        else:
            # Generate a key findings section from other content
            report += "## Key Findings\n\n"
            findings = []
            for section in sections:
                if 'key_points' in section:
                    findings.extend(section['key_points'])
            
            if findings:
                for i, finding in enumerate(findings):
                    report += f"{i+1}. {finding}\n"
            else:
                report += "*Key findings are summarized in the respective sections of this report.*\n"
            
            report += "\n"
        
        # Add other content sections
        other_sections = [s for s in sections if s.get("type") not in [
            "abstract", "executive_summary", "introduction", "key_findings", 
            "recommendations", "conclusion", "references"
        ]]
        
        for section in other_sections:
            report += f"## {section['title']}\n\n"
            
            # Add content with citations
            content = section['content']
            
            # Process citations
            if 'claims' in section:
                for claim in section['claims']:
                    if 'citation_id' in claim and claim['citation_id'] < len(citations):
                        citation = citations[claim['citation_id']]
                        content = content.replace(
                            claim['text'], 
                            f"{claim['text']} {citation['in_text_citation']}"
                        )
            
            report += f"{content}\n\n"
            
            # Add subsections
            if 'subsections' in section:
                for subsection in section['subsections']:
                    report += f"### {subsection['title']}\n\n{subsection['content']}\n\n"
        
        # Add recommendations section
        rec_section = next((s for s in sections if s.get("type") == "recommendations"), None)
        if rec_section:
            report += f"## Recommendations\n\n{rec_section['content']}\n\n"
        
        # Add conclusion
        conclusion_section = next((s for s in sections if s.get("type") == "conclusion"), None)
        if conclusion_section:
            report += f"## Conclusion\n\n{conclusion_section['content']}\n\n"
        
        # Add references
        report += "## References\n\n"
        
        unique_references = {}
        for citation in citations:
            if 'reference' in citation:
                unique_references[citation['reference']] = True
        
        for reference in sorted(unique_references.keys()):
            report += f"{reference}\n\n"
        
        # Add contact information if present
        if 'contact' in metadata:
            report += "---\n\n"
            report += f"**Contact Information:** {metadata['contact']}\n"
        
        return report
    
    # Technical template implementations
    def _technical_markdown_template(self, 
                                   title: str, 
                                   sections: List[Dict[str, Any]], 
                                   citations: List[Dict[str, Any]],
                                   metadata: Dict[str, Any]) -> str:
        """Generate a technical research report in Markdown format."""
        # Technical reports typically include:
        # - Abstract
        # - Introduction
        # - System Architecture
        # - Technical Specifications
        # - Implementation Details
        # - Performance Evaluation
        # - Results and Analysis
        # - Conclusion
        # - References
        # - Appendices (code samples, diagrams, etc.)
        
        report = f"# {title}\n\n"
        
        # Add metadata
        report += f"**Author:** {metadata.get('author', 'Research Agent')}\n"
        report += f"**Version:** {metadata.get('version', '1.0')}\n"
        report += f"**Date:** {metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}\n\n"
        
        # Add table of contents
        report += "## Table of Contents\n\n"
        
        # Add abstract if present
        abstract_section = next((s for s in sections if s.get("type") == "abstract"), None)
        if abstract_section:
            report += f"## Abstract\n\n{abstract_section['content']}\n\n"
        
        # Add introduction
        intro_section = next((s for s in sections if s.get("type") == "introduction"), None)
        if intro_section:
            report += f"## Introduction\n\n{intro_section['content']}\n\n"
        
        # Add system architecture if present
        arch_section = next((s for s in sections if s.get("type") == "architecture"), None)
        if arch_section:
            report += f"## System Architecture\n\n{arch_section['content']}\n\n"
        
        # Add technical specifications if present
        spec_section = next((s for s in sections if s.get("type") == "specifications"), None)
        if spec_section:
            report += f"## Technical Specifications\n\n{spec_section['content']}\n\n"
        
        # Add other content sections
        other_sections = [s for s in sections if s.get("type") not in [
            "abstract", "introduction", "architecture", "specifications", 
            "evaluation", "results", "conclusion", "references"
        ]]
        
        for section in other_sections:
            report += f"## {section['title']}\n\n"
            
            # Add content with citations
            content = section['content']
            
            # Process citations
            if 'claims' in section:
                for claim in section['claims']:
                    if 'citation_id' in claim and claim['citation_id'] < len(citations):
                        citation = citations[claim['citation_id']]
                        content = content.replace(
                            claim['text'], 
                            f"{claim['text']} {citation['in_text_citation']}"
                        )
            
            report += f"{content}\n\n"
            
            # Add subsections
            if 'subsections' in section:
                for subsection in section['subsections']:
                    report += f"### {subsection['title']}\n\n{subsection['content']}\n\n"
        
        # Add evaluation section if present
        eval_section = next((s for s in sections if s.get("type") == "evaluation"), None)
        if eval_section:
            report += f"## Performance Evaluation\n\n{eval_section['content']}\n\n"
        
        # Add results section if present
        results_section = next((s for s in sections if s.get("type") == "results"), None)
        if results_section:
            report += f"## Results and Analysis\n\n{results_section['content']}\n\n"
        
        # Add conclusion
        conclusion_section = next((s for s in sections if s.get("type") == "conclusion"), None)
        if conclusion_section:
            report += f"## Conclusion\n\n{conclusion_section['content']}\n\n"
        
        # Add references
        report += "## References\n\n"
        
        unique_references = {}
        for citation in citations:
            if 'reference' in citation:
                unique_references[citation['reference']] = True
        
        for reference in sorted(unique_references.keys()):
            report += f"{reference}\n\n"
        
        # Add code samples if present in metadata
        if 'code_samples' in metadata:
            report += "## Code Samples\n\n"
            code_samples = metadata['code_samples']
            if isinstance(code_samples, list):
                for i, sample in enumerate(code_samples):
                    if isinstance(sample, dict) and 'language' in sample and 'code' in sample:
                        report += f"### Sample {i+1}\n\n"
                        if 'description' in sample:
                            report += f"{sample['description']}\n\n"
                        report += f"```{sample['language']}\n{sample['code']}\n```\n\n"
        
        return report
    
    # Add placeholder implementations for HTML and text formats
    def _academic_html_template(self, title, sections, citations, metadata):
        """Placeholder for academic HTML template"""
        return self._default_html_template(title, sections, citations, metadata)
    
    def _business_html_template(self, title, sections, citations, metadata):
        """Placeholder for business HTML template"""
        return self._default_html_template(title, sections, citations, metadata)
    
    def _technical_html_template(self, title, sections, citations, metadata):
        """Placeholder for technical HTML template"""
        return self._default_html_template(title, sections, citations, metadata)
    
    def _academic_text_template(self, title, sections, citations, metadata):
        """Placeholder for academic text template"""
        return self._default_text_template(title, sections, citations, metadata)
    
    def _business_text_template(self, title, sections, citations, metadata):
        """Placeholder for business text template"""
        return self._default_text_template(title, sections, citations, metadata)
    
    def _technical_text_template(self, title, sections, citations, metadata):
        """Placeholder for technical text template"""
        return self._default_text_template(title, sections, citations, metadata)
    
    def _academic_latex_template(self, title, sections, citations, metadata):
        """Placeholder for academic LaTeX template"""
        return self._default_latex_template(title, sections, citations, metadata)
    
    def _business_latex_template(self, title, sections, citations, metadata):
        """Placeholder for business LaTeX template"""
        return self._default_latex_template(title, sections, citations, metadata)
    
    def _technical_latex_template(self, title, sections, citations, metadata):
        """Placeholder for technical LaTeX template"""
        return self._default_latex_template(title, sections, citations, metadata)
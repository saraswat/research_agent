"""
Unit tests for CitationGenerationTool.
"""
import unittest
from typing import Dict, List, Any

from tools.citation_generation_tool import CitationGenerationTool


class TestCitationGenerationTool(unittest.TestCase):
    """Test cases for the CitationGenerationTool."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.citation_tool = CitationGenerationTool()
        
        # Sample sources for testing
        self.web_source = {
            "title": "Advances in AI Research",
            "url": "https://example.com/ai-research",
            "author": "Smith, J.",
            "site_name": "AI Research Journal",
            "published_date": "2023-05-15"
        }
        
        self.book_source = {
            "title": "Machine Learning Fundamentals",
            "author": "Johnson, A.",
            "year": "2020",
            "publisher": "Tech Press",
            "location": "New York"
        }
        
        self.journal_source = {
            "title": "Neural Networks in Practice",
            "author": "Williams, R.",
            "year": "2022",
            "journal": "Journal of Machine Learning",
            "volume": "12",
            "issue": "3",
            "pages": "45-67",
            "doi": "10.1234/jml.2022.12345"
        }
        
        self.sample_claim = "Neural networks have shown significant improvements in image recognition tasks."
        self.sample_content = """
        Recent studies have demonstrated that neural networks have shown significant 
        improvements in image recognition tasks. The accuracy rates have improved by
        over 25% in the last five years, with convolutional neural networks leading
        the advances in this field. Researchers at major universities have confirmed
        these findings across multiple datasets.
        """

    def test_generate_web_citation_apa(self) -> None:
        """Test generating an APA citation for a web source."""
        self.citation_tool.set_citation_style("APA")
        citation = self.citation_tool.generate_citation(self.web_source)
        
        self.assertEqual(citation["source_type"], "web")
        self.assertEqual(citation["style"], "APA")
        self.assertIn("Smith, J.", citation["in_text_citation"])
        self.assertIn("2023", citation["in_text_citation"])
        self.assertIn("Smith, J.", citation["reference"])
        self.assertIn("Advances in AI Research", citation["reference"])
        self.assertIn("AI Research Journal", citation["reference"])
        self.assertIn("https://example.com/ai-research", citation["reference"])

    def test_generate_book_citation_mla(self) -> None:
        """Test generating an MLA citation for a book source."""
        self.citation_tool.set_citation_style("MLA")
        citation = self.citation_tool.generate_citation(self.book_source)
        
        self.assertEqual(citation["source_type"], "book")
        self.assertEqual(citation["style"], "MLA")
        self.assertIn("Johnson", citation["in_text_citation"])
        self.assertIn("Johnson, A.", citation["reference"])
        self.assertIn("Machine Learning Fundamentals", citation["reference"])
        self.assertIn("Tech Press", citation["reference"])
        self.assertIn("2020", citation["reference"])

    def test_generate_journal_citation_chicago(self) -> None:
        """Test generating a Chicago citation for a journal source."""
        self.citation_tool.set_citation_style("Chicago")
        citation = self.citation_tool.generate_citation(self.journal_source)
        
        self.assertEqual(citation["source_type"], "journal")
        self.assertEqual(citation["style"], "CHICAGO")
        self.assertIn("Williams", citation["in_text_citation"])
        self.assertIn("2022", citation["in_text_citation"])
        self.assertIn("Williams, R.", citation["reference"])
        self.assertIn("Neural Networks in Practice", citation["reference"])
        self.assertIn("Journal of Machine Learning", citation["reference"])
        self.assertIn("12, no. 3", citation["reference"])
        self.assertIn("45-67", citation["reference"])

    def test_generate_citation_ieee(self) -> None:
        """Test generating an IEEE citation."""
        self.citation_tool.set_citation_style("IEEE")
        citation = self.citation_tool.generate_citation(self.journal_source)
        
        self.assertEqual(citation["style"], "IEEE")
        self.assertIn("[", citation["in_text_citation"])
        self.assertIn("]", citation["in_text_citation"])
        self.assertIn("Williams, R.", citation["reference"])
        self.assertIn("Neural Networks in Practice", citation["reference"])
        self.assertIn("Journal of Machine Learning", citation["reference"])

    def test_verify_claim(self) -> None:
        """Test claim verification against source content."""
        verification = self.citation_tool.verify_claim(
            self.sample_claim, 
            self.sample_content
        )
        
        self.assertTrue(verification["supported"])
        self.assertGreater(verification["confidence"], 0.6)
        
        # Test with a claim that doesn't match
        unrelated_claim = "Quantum computing has revolutionized cryptography algorithms."
        unrelated_verification = self.citation_tool.verify_claim(
            unrelated_claim, 
            self.sample_content
        )
        
        self.assertFalse(unrelated_verification["supported"])
        self.assertLess(unrelated_verification["confidence"], 0.5)

    def test_citation_deduplication(self) -> None:
        """Test that duplicate sources are handled correctly."""
        # Generate citation for a source
        first_citation = self.citation_tool.generate_citation(self.web_source)
        
        # Generate citation for the same source again
        duplicate_source = {
            "title": "Advances in AI Research",
            "url": "https://example.com/ai-research",
            "author": "Smith, J.",
            "site_name": "AI Research Journal",
            "published_date": "2023-05-15"
        }
        second_citation = self.citation_tool.generate_citation(duplicate_source)
        
        # The citations should have the same ID
        self.assertEqual(first_citation["id"], second_citation["id"])
        
        # Generate bibliography and check there's only one entry
        bibliography = self.citation_tool.generate_bibliography()
        self.assertEqual(bibliography["count"], 1)

    def test_citation_with_claim(self) -> None:
        """Test associating citations with claims."""
        # Generate a citation and associate it with a claim
        citation = self.citation_tool.generate_citation(self.web_source, self.sample_claim)
        
        # Retrieve citations for the claim
        claim_citations = self.citation_tool.get_citation_for_claim(self.sample_claim)
        
        self.assertEqual(len(claim_citations), 1)
        self.assertEqual(claim_citations[0]["id"], citation["id"])

    def test_manage_citation_versions(self) -> None:
        """Test updating citation versions."""
        # Generate a citation
        citation = self.citation_tool.generate_citation(self.web_source)
        
        # Update the citation with new information
        updated_data = {"published_date": "2023-06-20"}
        result = self.citation_tool.manage_citation_versions(citation["id"], updated_data)
        
        self.assertTrue(result["success"])
        
        # Get the updated citation from the database
        updated_citation = self.citation_tool.citation_db[citation["id"]]
        self.assertEqual(updated_citation["source_data"]["published_date"], "2023-06-20")
        
        # Verify reference was updated
        self.assertIn("2023", updated_citation["reference"])

    def test_export_citations(self) -> None:
        """Test exporting citations in different formats."""
        # Add a few citations
        self.citation_tool.generate_citation(self.web_source)
        self.citation_tool.generate_citation(self.book_source)
        self.citation_tool.generate_citation(self.journal_source)
        
        # Test plain text export
        plain_export = self.citation_tool.export_citations("plain")
        self.assertTrue(plain_export["success"])
        self.assertEqual(plain_export["format"], "plain")
        self.assertIn("Smith, J.", plain_export["content"])
        
        # Test HTML export
        html_export = self.citation_tool.export_citations("html")
        self.assertTrue(html_export["success"])
        self.assertEqual(html_export["format"], "html")
        self.assertIn("<li>", html_export["content"])
        
        # Test BibTeX export
        bibtex_export = self.citation_tool.export_citations("bibtex")
        self.assertTrue(bibtex_export["success"])
        self.assertEqual(bibtex_export["format"], "bibtex")
        self.assertIn("@", bibtex_export["content"])
        self.assertIn("author", bibtex_export["content"])
        self.assertIn("title", bibtex_export["content"])


if __name__ == "__main__":
    unittest.main()
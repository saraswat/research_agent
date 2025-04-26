"""
Unit tests for InformationExtractionTool.
"""
import unittest
from typing import Dict, List, Any

from tools.information_extraction_tool import InformationExtractionTool


class TestInformationExtractionTool(unittest.TestCase):
    """Test cases for the InformationExtractionTool."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.extraction_tool = InformationExtractionTool()
        self.sample_text = """
        In a recent study conducted by Stanford University, researchers found that 75% of participants 
        showed improved cognitive function after following the new treatment protocol. 
        The study, which involved 500 participants aged 65 to 85, demonstrated a significant reduction 
        in cognitive decline over a period of 24 months.
        
        Dr. Sarah Johnson, the lead researcher, stated that "these findings represent a major 
        breakthrough in our understanding of cognitive aging processes." She further explained that 
        the treatment combines both pharmaceutical and lifestyle interventions.
        
        According to the World Health Organization, approximately 50 million people worldwide 
        are living with dementia, with nearly 10 million new cases every year.
        
        The cost of treatment is estimated at $5,000 per patient annually, making it accessible 
        for many healthcare systems. In contrast, traditional treatments often cost between $8,000 to $12,000 per year.
        
        Cognitive aging is defined as the normal process of cognitive change that occurs as people get older.
        Unlike dementia, it is not considered a disease but rather a natural part of aging.
        
        The research team, which includes experts from Harvard Medical School and Mayo Clinic, 
        plans to expand the study to include younger participants starting in January 2023.
        
        For more information, contact research@stanford.edu or visit www.cognitive-study.org.
        """

    def test_extract_statistics(self) -> None:
        """Test extracting statistical information."""
        result = self.extraction_tool.extract(
            self.sample_text, 
            "Extract statistics about participant percentages"
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["extraction_type"], "statistics")
        
        # Should find at least one percentage statistic
        stats = result["extracted_information"]
        self.assertGreater(len(stats), 0)
        
        # Check if 75% is found
        percentage_found = False
        for stat in stats:
            if stat["type"] == "percentage" and stat["value"] == "75%":
                percentage_found = True
                break
        
        self.assertTrue(percentage_found, "Failed to extract 75% statistic")

    def test_extract_quotes(self) -> None:
        """Test extracting quotes."""
        result = self.extraction_tool.extract(
            self.sample_text, 
            "Extract quotes from Dr. Johnson"
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["extraction_type"], "quotes")
        
        quotes = result["extracted_information"]
        self.assertGreater(len(quotes), 0)
        
        # Check if the direct quote is found
        quote_found = False
        for quote in quotes:
            if "breakthrough" in quote.get("quote", ""):
                quote_found = True
                break
        
        self.assertTrue(quote_found, "Failed to extract Dr. Johnson's quote")

    def test_extract_definitions(self) -> None:
        """Test extracting definitions."""
        result = self.extraction_tool.extract(
            self.sample_text, 
            "Find the definition of cognitive aging"
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["extraction_type"], "definitions")
        
        definitions = result["extracted_information"]
        self.assertGreater(len(definitions), 0)
        
        # Check if the definition is found
        definition_found = False
        for definition in definitions:
            if "term" in definition and definition["term"] == "cognitive aging":
                definition_found = True
                break
        
        self.assertTrue(definition_found, "Failed to extract definition of cognitive aging")

    def test_extract_entities(self) -> None:
        """Test extracting named entities."""
        result = self.extraction_tool.extract(
            self.sample_text, 
            "Identify organizations mentioned in the text"
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["extraction_type"], "entities")
        
        entities = result["extracted_information"]
        self.assertGreater(len(entities), 0)
        
        # Check if Stanford University is found
        stanford_found = False
        who_found = False
        
        for entity in entities:
            if entity.get("type") == "organization":
                if "Stanford" in entity.get("entity", ""):
                    stanford_found = True
                if "World Health Organization" in entity.get("entity", "") or "WHO" in entity.get("entity", ""):
                    who_found = True
        
        self.assertTrue(stanford_found, "Failed to extract Stanford University as an organization")
        self.assertTrue(who_found, "Failed to extract World Health Organization as an organization")

    def test_general_extraction(self) -> None:
        """Test general information extraction."""
        result = self.extraction_tool.extract(
            self.sample_text, 
            "Find information about the research team"
        )
        
        self.assertTrue(result["success"])
        
        extracted_info = result["extracted_information"]
        self.assertGreater(len(extracted_info), 0)
        
        # At least one extracted piece should mention Harvard or Mayo Clinic
        research_team_found = False
        for info in extracted_info:
            if "Harvard" in info.get("content", "") or "Mayo" in info.get("content", ""):
                research_team_found = True
                break
        
        self.assertTrue(research_team_found, "Failed to extract information about research team")

    def test_currency_extraction(self) -> None:
        """Test extracting currency values."""
        result = self.extraction_tool.extract(
            self.sample_text, 
            "Extract cost information"
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["extraction_type"], "statistics")
        
        stats = result["extracted_information"]
        
        # Should find the $5,000 cost
        cost_found = False
        for stat in stats:
            if stat["type"] == "currency" and "$5,000" in stat["value"]:
                cost_found = True
                break
        
        self.assertTrue(cost_found, "Failed to extract $5,000 cost information")
        
    def test_date_extraction(self) -> None:
        """Test extracting dates."""
        result = self.extraction_tool.extract(
            self.sample_text, 
            "Extract dates mentioned"
        )
        
        self.assertTrue(result["success"])
        extracted_info = result["extracted_information"]
        
        # Should find January 2023
        date_found = False
        if result["extraction_type"] == "entities":
            for entity in extracted_info:
                if entity.get("type") == "date" and "2023" in entity.get("entity", ""):
                    date_found = True
                    break
        
        self.assertTrue(date_found, "Failed to extract January 2023 date")


    def test_fact_checking(self) -> None:
        """Test fact checking functionality."""
        # This test will only check if the fact checking interface works properly
        # We won't actually check external sources in a unit test
        
        result = self.extraction_tool.extract(
            self.sample_text, 
            "Identify organizations and verify them",  # The "verify" keyword triggers fact checking
            fact_check=True  # Explicitly request fact checking
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(result["fact_checked"])  # Should indicate fact checking was performed
        
        # Check if at least one entity has verification information
        entities = result["extracted_information"]
        verification_found = False
        
        for entity in entities:
            if entity.get("verification_method") is not None:
                verification_found = True
                break
        
        self.assertTrue(verification_found, "Failed to add verification information to entities")


if __name__ == "__main__":
    unittest.main()
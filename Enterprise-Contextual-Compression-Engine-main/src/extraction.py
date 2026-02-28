"""
Fact Extraction Module

Extracts decision-critical facts including:
- numbers
- thresholds
- limits
- risks
- constraints
- exceptions
- compliance rules

Enhanced with confidence scoring and improved pattern matching.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactExtractor:
    """
    Extracts decision-critical facts from text with confidence scoring.
    """
    
    def __init__(self):
        """Initialize the fact extractor with enhanced patterns."""
        # Enhanced pattern for numbers (currency, percentages, decimals, scientific notation)
        self.number_pattern = re.compile(
            r'(?:^|\s)(?:USD\s*)?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+%?|\d+%'
        )
        
        # Pattern for currency amounts with better precision
        self.currency_pattern = re.compile(
            r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars?)?'
        )
        
        # Pattern for percentages
        self.percentage_pattern = re.compile(r'\d+(?:\.\d+)?%')
        
        # Pattern for time periods (days, hours, years, etc.)
        self.time_pattern = re.compile(
            r'\d+\s*(?:days?|hours?|minutes?|weeks?|months?|years?|business\s+days?)'
        )
        
        # Enhanced keywords for thresholds and limits (with confidence weights)
        self.threshold_keywords = {
            'threshold': 1.0, 'limit': 1.0, 'maximum': 1.0, 'minimum': 1.0,
            'max': 0.9, 'min': 0.9, 'cap': 0.95, 'ceiling': 0.95,
            'floor': 0.9, 'bound': 0.85, 'boundary': 0.85,
            'up to': 0.9, 'not exceeding': 0.95, 'not more than': 0.95,
            'at least': 0.9, 'at most': 0.9, 'cannot exceed': 0.95,
            'must not exceed': 0.95, 'limited to': 0.9
        }
        
        # Enhanced keywords for risks (with confidence weights)
        self.risk_keywords = {
            'risk': 1.0, 'hazard': 0.95, 'danger': 0.95, 'warning': 0.9,
            'caution': 0.85, 'adverse': 0.9, 'negative': 0.8,
            'failure': 0.9, 'error': 0.85, 'fault': 0.85, 'issue': 0.8,
            'problem': 0.8, 'threat': 0.95, 'vulnerability': 0.9,
            'exposure': 0.85, 'suspicious': 0.9, 'fraud': 0.95,
            'breach': 0.9, 'violation': 0.9
        }
        
        # Enhanced keywords for constraints (with confidence weights)
        self.constraint_keywords = {
            'constraint': 1.0, 'restriction': 0.95, 'limitation': 0.95,
            'requirement': 0.9, 'must': 0.95, 'shall': 0.95, 'should': 0.85,
            'required': 0.9, 'mandatory': 0.95, 'prohibited': 0.95,
            'forbidden': 0.95, 'not allowed': 0.9, 'cannot': 0.9,
            'must not': 0.95, 'shall not': 0.95, 'strictly': 0.85
        }
        
        # Enhanced keywords for exceptions (with confidence weights)
        self.exception_keywords = {
            'except': 0.95, 'exception': 1.0, 'unless': 0.95,
            'excluding': 0.9, 'exclude': 0.9, 'however': 0.85,
            'but': 0.7, 'alternatively': 0.85, 'otherwise': 0.85,
            'unless otherwise': 0.95, 'waiver': 0.9, 'waived': 0.9,
            'special circumstances': 0.9
        }
        
        # Enhanced keywords for compliance rules (with confidence weights)
        self.compliance_keywords = {
            'compliance': 1.0, 'regulation': 0.95, 'standard': 0.9,
            'policy': 0.85, 'rule': 0.85, 'guideline': 0.85,
            'protocol': 0.9, 'procedure': 0.85, 'must comply': 0.95,
            'in accordance with': 0.9, 'per': 0.8, 'as per': 0.85,
            'according to': 0.85, 'regulatory': 0.9, 'audit': 0.85,
            'compliance officer': 0.9
        }
    
    def extract_facts(
        self,
        paragraph: Dict[str, str],
        section_id: str,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract decision-critical facts from a paragraph with confidence scores.
        
        Args:
            paragraph: Dictionary with 'paragraph_id' and 'text'
            section_id: ID of the section containing this paragraph
            document_id: ID of the document
            
        Returns:
            List of extracted facts, each with:
            {
                'fact_text': str,
                'fact_type': str,
                'confidence_score': float,  # 0.0-1.0
                'document_id': str,
                'section_id': str,
                'paragraph_id': str,
                'original_text': str
            }
        """
        text = paragraph['text']
        paragraph_id = paragraph['paragraph_id']
        
        facts = []
        
        # Extract facts with numbers (thresholds, limits, etc.) - highest priority
        number_facts = self._extract_number_facts(
            text, document_id, section_id, paragraph_id
        )
        facts.extend(number_facts)
        
        # Extract risk-related facts
        risk_facts = self._extract_keyword_facts(
            text, document_id, section_id, paragraph_id,
            self.risk_keywords, 'risk'
        )
        facts.extend(risk_facts)
        
        # Extract constraint-related facts
        constraint_facts = self._extract_keyword_facts(
            text, document_id, section_id, paragraph_id,
            self.constraint_keywords, 'constraint'
        )
        facts.extend(constraint_facts)
        
        # Extract exception-related facts
        exception_facts = self._extract_keyword_facts(
            text, document_id, section_id, paragraph_id,
            self.exception_keywords, 'exception'
        )
        facts.extend(exception_facts)
        
        # Extract compliance-related facts
        compliance_facts = self._extract_keyword_facts(
            text, document_id, section_id, paragraph_id,
            self.compliance_keywords, 'compliance'
        )
        facts.extend(compliance_facts)
        
        # Remove duplicates and merge confidence scores
        unique_facts = self._deduplicate_facts(facts)
        
        logger.debug(
            f"Extracted {len(unique_facts)} facts from paragraph {paragraph_id}"
        )
        
        return unique_facts
    
    def _deduplicate_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate facts and merge confidence scores.
        
        Args:
            facts: List of fact dictionaries
            
        Returns:
            List of unique facts with merged confidence scores
        """
        fact_map = {}
        
        for fact in facts:
            fact_key = fact['fact_text'].lower().strip()
            
            if fact_key in fact_map:
                # Merge: take maximum confidence and combine fact types
                existing = fact_map[fact_key]
                existing['confidence_score'] = max(
                    existing.get('confidence_score', 0.0),
                    fact.get('confidence_score', 0.0)
                )
                # Combine fact types if different
                if fact['fact_type'] != existing['fact_type']:
                    existing['fact_type'] = f"{existing['fact_type']},{fact['fact_type']}"
            else:
                fact_map[fact_key] = fact.copy()
        
        return list(fact_map.values())
    
    def _extract_number_facts(
        self,
        text: str,
        document_id: str,
        section_id: str,
        paragraph_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract facts containing numbers (thresholds, limits, etc.) with confidence scoring.
        
        Args:
            text: Text to extract from
            document_id: Document ID
            section_id: Section ID
            paragraph_id: Paragraph ID
            
        Returns:
            List of fact dictionaries with confidence scores
        """
        facts = []
        
        # Split into sentences (handle multiple sentence endings)
        sentences = re.split(r'[.!?]+(?:\s+|$)', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check if sentence contains numbers
            has_number = bool(self.number_pattern.search(sentence))
            has_currency = bool(self.currency_pattern.search(sentence))
            has_percentage = bool(self.percentage_pattern.search(sentence))
            has_time = bool(self.time_pattern.search(sentence))
            
            if not (has_number or has_currency or has_percentage or has_time):
                continue
            
            sentence_lower = sentence.lower()
            
            # Calculate confidence based on indicators
            confidence = 0.5  # Base confidence for having a number
            
            # Boost confidence for threshold/limit keywords
            threshold_matches = [
                keyword for keyword, weight in self.threshold_keywords.items()
                if keyword in sentence_lower
            ]
            if threshold_matches:
                max_weight = max(self.threshold_keywords[kw] for kw in threshold_matches)
                confidence = max(confidence, max_weight)
                fact_type = 'threshold'
            else:
                fact_type = 'number'
            
            # Boost confidence for specific patterns
            if has_currency:
                confidence = min(1.0, confidence + 0.2)
            if has_percentage:
                confidence = min(1.0, confidence + 0.15)
            if has_time:
                confidence = min(1.0, confidence + 0.1)
            
            # Boost for multiple numbers (indicates specific threshold)
            number_count = len(self.number_pattern.findall(sentence))
            if number_count > 1:
                confidence = min(1.0, confidence + 0.1)
            
            # Boost for comparison words (greater than, less than, etc.)
            comparison_words = ['exceed', 'above', 'below', 'between', 'within', 'at least', 'at most']
            if any(word in sentence_lower for word in comparison_words):
                confidence = min(1.0, confidence + 0.15)
            
            # Only include facts with reasonable confidence
            if confidence >= 0.4:
                facts.append({
                    'fact_text': sentence.strip(),
                    'fact_type': fact_type,
                    'confidence_score': round(confidence, 3),
                    'document_id': document_id,
                    'section_id': section_id,
                    'paragraph_id': paragraph_id,
                    'original_text': text
                })
        
        return facts
    
    def _extract_keyword_facts(
        self,
        text: str,
        document_id: str,
        section_id: str,
        paragraph_id: str,
        keywords: Dict[str, float],
        fact_type: str
    ) -> List[Dict[str, Any]]:
        """
        Extract facts containing specific keywords with confidence scoring.
        
        Args:
            text: Text to extract from
            document_id: Document ID
            section_id: Section ID
            paragraph_id: Paragraph ID
            keywords: Dictionary mapping keywords to confidence weights
            fact_type: Type of fact (risk, constraint, etc.)
            
        Returns:
            List of fact dictionaries with confidence scores
        """
        facts = []
        text_lower = text.lower()
        
        # Find matching keywords with their weights
        found_keywords = {
            keyword: weight
            for keyword, weight in keywords.items()
            if keyword in text_lower
        }
        
        if found_keywords:
            # Split into sentences
            sentences = re.split(r'[.!?]+(?:\s+|$)', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                sentence_lower = sentence.lower()
                
                # Find keywords in this sentence
                sentence_keywords = {
                    kw: weight for kw, weight in found_keywords.items()
                    if kw in sentence_lower
                }
                
                if sentence_keywords:
                    # Confidence is based on highest-weight keyword found
                    max_confidence = max(sentence_keywords.values())
                    
                    # Boost confidence if multiple keywords found
                    if len(sentence_keywords) > 1:
                        max_confidence = min(1.0, max_confidence + 0.1)
                    
                    # Boost confidence if sentence contains numbers
                    if self.number_pattern.search(sentence):
                        max_confidence = min(1.0, max_confidence + 0.15)
                    
                    # Only include facts with reasonable confidence
                    if max_confidence >= 0.5:
                        facts.append({
                            'fact_text': sentence.strip(),
                            'fact_type': fact_type,
                            'confidence_score': round(max_confidence, 3),
                            'document_id': document_id,
                            'section_id': section_id,
                            'paragraph_id': paragraph_id,
                            'original_text': text
                        })
        
        return facts

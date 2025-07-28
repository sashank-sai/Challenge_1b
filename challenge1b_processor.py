#!/usr/bin/env python3
"""
Adobe Hackathon 2025 - Challenge 1b: Multi-Collection PDF Analysis
Persona-based content extraction and importance ranking system
"""

import json
import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import logging
from dataclasses import dataclass
import math
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PersonaConfig:
    """Configuration for different personas and their preferences"""
    role: str
    keywords: List[str]
    priority_sections: List[str]
    importance_weights: Dict[str, float]

class PersonaAnalyzer:
    """Analyzes content relevance based on user personas"""
    
    def __init__(self):
        self.personas = {
            "Travel Planner": PersonaConfig(
                role="Travel Planner",
                keywords=["itinerary", "attractions", "accommodation", "restaurants", "transportation", 
                         "budget", "activities", "sightseeing", "hotels", "booking", "tickets", "schedule"],
                priority_sections=["attractions", "restaurants", "hotels", "activities", "transportation", "itinerary"],
                importance_weights={
                    "attractions": 1.0,
                    "restaurants": 0.9,
                    "hotels": 0.9,
                    "activities": 0.8,
                    "transportation": 0.8,
                    "budget": 0.7,
                    "itinerary": 1.0
                }
            ),
            "HR Professional": PersonaConfig(
                role="HR Professional",
                keywords=["forms", "fillable", "onboarding", "compliance", "employee", "documents", 
                         "fields", "signatures", "workflow", "templates", "automation", "policies"],
                priority_sections=["forms", "templates", "workflow", "compliance", "onboarding", "automation"],
                importance_weights={
                    "forms": 1.0,
                    "fillable": 1.0,
                    "compliance": 0.9,
                    "onboarding": 0.9,
                    "templates": 0.8,
                    "workflow": 0.8,
                    "automation": 0.7
                }
            ),
            "Food Contractor": PersonaConfig(
                role="Food Contractor",
                keywords=["vegetarian", "buffet", "catering", "menu", "recipes", "ingredients", 
                         "portions", "dietary", "cooking", "preparation", "corporate", "dinner", "gluten-free"],
                priority_sections=["recipes", "vegetarian", "buffet", "catering", "menu", "ingredients"],
                importance_weights={
                    "vegetarian": 1.0,
                    "buffet": 1.0,
                    "catering": 0.9,
                    "recipes": 0.9,
                    "menu": 0.8,
                    "ingredients": 0.8,
                    "portions": 0.7,
                    "gluten-free": 0.9
                }
            )
        }
    
    def get_persona_config(self, persona_role: str) -> PersonaConfig:
        """Get configuration for a specific persona"""
        return self.personas.get(persona_role, self.personas["Travel Planner"])
    
    def calculate_relevance_score(self, text: str, persona: PersonaConfig, job_description: str) -> float:
        """Calculate relevance score for text based on persona and job"""
        text_lower = text.lower()
        job_lower = job_description.lower()
        
        # Base score from persona keywords
        keyword_score = 0
        for keyword in persona.keywords:
            if keyword in text_lower:
                keyword_score += 1
        
        # Job-specific keywords boost
        job_words = set(job_lower.split())
        text_words = set(text_lower.split())
        job_overlap = len(job_words.intersection(text_words))
        
        # Priority section bonus
        section_bonus = 0
        for priority_section in persona.priority_sections:
            if priority_section in text_lower:
                weight = persona.importance_weights.get(priority_section, 0.5)
                section_bonus += weight
        
        # Combine scores
        total_score = (keyword_score * 0.4) + (job_overlap * 0.3) + (section_bonus * 0.3)
        
        # Normalize by text length (prefer concise, relevant content)
        length_factor = min(1.0, 100 / max(len(text.split()), 1))
        
        return total_score * length_factor

class MultiCollectionProcessor:
    """Main processor for multi-collection PDF analysis"""
    
    def __init__(self, collection_path: Path):
        self.collection_path = collection_path
        self.persona_analyzer = PersonaAnalyzer()
        # Font size thresholds for section classification
        self.title_font_threshold = 16
        self.heading_font_threshold = 14
        self.subheading_font_threshold = 12
        self.min_relevance_score = 0.1  # Define minimum relevance score
    
    def process_collection(self):
        """Process a single collection with its input configuration"""
        try:
            # Load input configuration
            input_config = self.load_input_config()
            if not input_config:
                logger.error(f"Failed to load input config for {self.collection_path}")
                return
            
            # Extract configuration details
            challenge_info = input_config.get("challenge_info", {})
            documents = input_config.get("documents", [])
            persona_info = input_config.get("persona", {})
            job_info = input_config.get("job_to_be_done", {})
            
            persona_role = persona_info.get("role", "Travel Planner")
            job_description = job_info.get("task", "")
            
            logger.info(f"Processing collection: {challenge_info.get('challenge_id', 'unknown')}")
            logger.info(f"Persona: {persona_role}")
            logger.info(f"Job: {job_description}")
            
            # Get persona configuration
            persona_config = self.persona_analyzer.get_persona_config(persona_role)
            
            # Process all PDFs in the collection
            all_sections = []
            all_subsections = []
            
            pdfs_path = self.collection_path / "PDFs"
            for doc_info in documents:
                pdf_filename = doc_info.get("filename", "")
                pdf_path = pdfs_path / pdf_filename
                
                if pdf_path.exists():
                    logger.info(f"Processing: {pdf_filename}")
                    sections, subsections = self.analyze_pdf(
                        pdf_path, persona_config, job_description
                    )
                    all_sections.extend(sections)
                    all_subsections.extend(subsections)
                else:
                    logger.warning(f"PDF not found: {pdf_path}")
            
            # Rank sections by importance
            all_sections = self.rank_sections_by_importance(all_sections, persona_config, job_description)
            
            # Generate output
            output_data = self.create_output_structure(
                input_config, all_sections, all_subsections
            )
            
            # Save output
            self.save_output(output_data)
            
            logger.info(f"Collection processing completed: {len(all_sections)} sections extracted")
            
        except Exception as e:
            logger.error(f"Error processing collection {self.collection_path}: {str(e)}")
    
    def load_input_config(self) -> Dict[str, Any]:
        """Load the input configuration JSON file"""
        input_file = self.collection_path / "challenge1b_input.json"
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading input config: {str(e)}")
            return {}
    
    def analyze_pdf(self, pdf_path: Path, persona: PersonaConfig, job: str) -> Tuple[List[Dict], List[Dict]]:
        """Analyze a PDF file and extract relevant sections and subsections"""
        sections = []
        subsections = []
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_sections = self.extract_page_sections(page, page_num + 1, filename)
            for section in page_sections:
                relevance_score = self.persona_analyzer.calculate_relevance_score(
                    section["content"], persona, job
                )
                if relevance_score >= self.min_relevance_score:
                    sections.append({
                        "document": filename,
                        "section_title": section["title"],
                        "importance_rank": 0,  # To be updated after sorting
                        "page_number": page_num + 1,
                        "relevance_score": relevance_score  # Store for sorting
                    })
                    subsections.append({
                        "document": filename,
                        "refined_text": self.refine_text_for_persona(section["content"], persona, job),
                        "page_number": page_num + 1
                    })
        doc.close()
        return sections, subsections
    
    def extract_page_sections(self, page: fitz.Page, page_num: int, filename: str) -> List[Dict[str, Any]]:
        """Extract sections from a single page"""
        sections = []
        
        # Get text blocks with formatting
        text_dict = page.get_text("dict")
        
        current_section = None
        
        for block in text_dict["blocks"]:
            if "lines" in block:  # Text block
                block_text = ""
                font_sizes = []
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_sizes.append(span["size"])
                    block_text += line_text + " "
                
                if block_text.strip():
                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                    text = block_text.strip()
                    
                    # Classify text type
                    if avg_font_size >= self.heading_font_threshold and len(text) < 200:
                        # This is likely a section heading
                        if current_section:
                            sections.append(current_section)
                        
                        current_section = {
                            "title": text,
                            "content": "",
                            "page": page_num,
                            "font_size": avg_font_size
                        }
                    elif current_section and avg_font_size < self.heading_font_threshold:
                        # This is content for the current section
                        current_section["content"] += text + " "
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        # If no sections found, create a general content section
        if not sections:
            full_text = page.get_text()
            if full_text.strip():
                sections.append({
                    "title": f"Content from {filename} - Page {page_num}",
                    "content": full_text.strip(),
                    "page": page_num,
                    "font_size": 12
                })
        
        return sections
    
    def rank_sections_by_importance(self, sections: List[Dict], persona_config: PersonaConfig, job_description: str) -> List[Dict]:
        """Rank sections by importance for the persona and job"""
        # Sort by relevance score (descending)
        sections.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Assign importance ranks
        for i, section in enumerate(sections):
            section["importance_rank"] = i + 1
            # Remove internal scoring fields from output
            section.pop("relevance_score", None)
            section.pop("content", None)  # Remove full content, keep only title for sections
        
        return sections
    
    def refine_text_for_persona(self, text: str, persona_config: PersonaConfig, job_description: str) -> str:
        """Refine and summarize text based on persona requirements"""
        # Extract sentences that are most relevant
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Check for persona keywords
            sentence_lower = sentence.lower()
            relevance = 0
            
            for keyword in persona_config.keywords:
                if keyword in sentence_lower:
                    relevance += 1
            
            # Check for job-specific terms
            job_words = set(job_description.lower().split())
            sentence_words = set(sentence_lower.split())
            job_overlap = len(job_words.intersection(sentence_words))
            
            if relevance > 0 or job_overlap > 1:
                relevant_sentences.append((sentence, relevance + job_overlap))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]  # Top 3 sentences
        
        refined_text = ". ".join(top_sentences)
        if refined_text and not refined_text.endswith('.'):
            refined_text += "."
        
        return refined_text or text[:200] + "..." if len(text) > 200 else text
    
    def create_output_structure(self, input_config: Dict, sections: List[Dict], subsections: List[Dict]) -> Dict[str, Any]:
        """Create the required output JSON structure"""
        documents = input_config.get("documents", [])
        persona_info = input_config.get("persona", {})
        job_info = input_config.get("job_to_be_done", {})
        
        output = {
            "metadata": {
                "input_documents": [doc.get("filename", "") for doc in documents],
                "persona": persona_info.get("role", ""),
                "job_to_be_done": job_info.get("task", ""),
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": sections,
            "subsection_analysis": subsections
        }
        
        return output
    
    def save_output(self, output_data: Dict[str, Any]):
        """Save the output JSON file to /app/output"""
        collection_name = self.collection_path.name
        output_file = Path("/app/output") / f"{collection_name}_challenge1b_output.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Output saved: {output_file}")
        except Exception as e:
            logger.error(f"Error saving output: {str(e)}")

def process_all_collections(base_path: Path):
    """Process all collections in the base directory"""
    logger.info("Starting multi-collection PDF analysis...")
    
    # Look for collection directories
    collection_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("Collection")]
    
    if not collection_dirs:
        logger.error("No collection directories found!")
        return
    
    logger.info(f"Found {len(collection_dirs)} collections to process")
    
    for collection_dir in sorted(collection_dirs):
        logger.info(f"\n=== Processing {collection_dir.name} ===")
        processor = MultiCollectionProcessor(collection_dir)
        processor.process_collection()
    
    logger.info("\nAll collections processed successfully!")

def main():
    """Main entry point"""
    # For Docker environment, look in /app/input
    # For local testing, look in current directory
    base_path = Path("/app/input") if Path("/app/input").exists() else Path(".")
    
    process_all_collections(base_path)

if __name__ == "__main__":
    main()
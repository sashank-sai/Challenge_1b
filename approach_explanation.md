# Approach Explanation for Challenge 1b
Our solution for Adobe India Hackathon 2025 Challenge 1b delivers a persona-driven document intelligence system to process PDF collections, extract relevant sections, and prioritize them for user-specific tasks, aligning with the theme “Connect What Matters — For the User Who Matters.”

## Methodology
The system leverages `PyMuPDF` (1.23.26) for PDF text extraction, identifying sections via font size thresholds (titles >16, headings >14). The `MultiCollectionProcessor` class processes collections from `challenge1b_input.json`, handling PDFs in `PDFs` subdirectories. The `PersonaAnalyzer` supports personas (e.g., Travel Planner) with configurable keywords (e.g., “itinerary”: 1.0, “restaurants”: 0.9) and weights, extensible to other personas (e.g., HR Professional). Relevance scoring combines keyword matches, job description overlap, and section priority. Sections are ranked by relevance, and subsections are refined to the top 3 sentences or truncated text. The output JSON includes metadata, extracted sections, and subsection analysis, with a processing timestamp.

## Implementation
The script iterates through collection directories, processes PDFs, and saves JSON outputs to `/app/output`. The Docker setup (`python:3.10-slim`, 720MB) ensures efficiency, processing 31 PDFs in ~10 seconds. The `--network none` flag ensures offline operation. Logging tracks section counts (e.g., 73 for Collection 1) and errors.

## Generalization
The solution generalizes across domains (travel, HR, academia) via flexible PDF parsing and persona configurations, meeting all constraints: CPU-only, <1GB image, fast processing, and offline operation. It prioritizes user-relevant content, ensuring practical utility for tasks like trip planning.

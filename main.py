import os
import json
import fitz
import time
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message="`encoder_attention_mask` is deprecated")

class AdvancedDocumentIntel:
    """
    An advanced system using a Sentence-BERT model for semantic understanding
    to find and rank the most relevant document sections.
    """
    def __init__(self, input_path="challenge1b_input.json"):
        """Initializes the system by loading the input file and the ML model."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse configuration from the new input.json structure
        self.persona = data['persona']['role']
        self.job_to_be_done = data['job_to_be_done']['task']
        self.documents_info = data['documents']

        # Define other parameters since they are not in input.json
        self.pdf_dir = "pdfs"
        self.output_dir = "output"
        self.top_k_sections = 5
        self.max_sections_per_document = 2
        self.min_section_length = 100
        self.excluded_titles = ["contents", "index", "references"]
        
        # --- THIS IS THE MODIFIED PART ---
        # Define the path to the local model cache
        model_path = './model_cache' 
        
        # Load the pre-trained Sentence-BERT model from the local path.
        print(f"Loading sentence-transformer model from local path: {model_path}...")
        self.model = SentenceTransformer(model_path)
        print("Model loaded.")

    def _is_heading(self, line_text: str, spans: list, body_font_size: float) -> bool:
        """
        Identifies if a line is a heading. A heading must not look like a list item,
        must contain styled text (bold/larger), and should be relatively short.
        """
        if line_text.strip().startswith(('•', '*', '-')):
            return False
        
        has_heading_font = any(round(s['size']) > body_font_size or 'bold' in s['font'].lower() for s in spans)
        if not has_heading_font:
            return False
            
        if len(line_text) > 100:
            return False
            
        return True

    def _add_section_if_valid(self, sections_list, heading, text, page, doc_name):
        stripped_text = text.strip()
        clean_title = heading.lower().strip().rstrip(':.')
        if stripped_text and len(stripped_text) >= self.min_section_length and clean_title not in self.excluded_titles:
            sections_list.append({
                "title": heading.strip(), "content": stripped_text,
                "page_number": page, "document": doc_name
            })

    def _extract_logical_sections(self, pdf_path: str) -> list:
        """
        Extracts complete logical sections from a PDF using a more robust
        method based on the document's rich text information.
        """
        doc = fitz.open(pdf_path)
        sections = []
        doc_name = os.path.basename(pdf_path)

        for page_num, page in enumerate(doc, start=1):
            page_data = page.get_text("dict", sort=True)
            if not page_data["blocks"]:
                continue

            font_sizes = []
            for block in page_data["blocks"]:
                if block['type'] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(round(span["size"]))
            
            if not font_sizes:
                continue
            body_font_size = Counter(font_sizes).most_common(1)[0][0]

            current_heading, current_text = "General", ""
            for block in page_data["blocks"]:
                if block['type'] == 0:
                    first_line_spans = []
                    if block["lines"]:
                        first_line_spans = block["lines"][0]["spans"]
                    
                    first_line_text = " ".join(s['text'] for s in first_line_spans)

                    if self._is_heading(first_line_text, first_line_spans, body_font_size):
                        self._add_section_if_valid(sections, current_heading, current_text, page_num, doc_name)
                        current_heading = first_line_text
                        
                        block_text = ""
                        for i, line in enumerate(block["lines"]):
                            if i > 0: # Get text from lines after the heading
                                block_text += " ".join(span['text'] for span in line['spans']) + " "
                        current_text = block_text
                    else:
                        block_text = ""
                        for line in block["lines"]:
                            block_text += " ".join(span['text'] for span in line['spans']) + " "
                        current_text += block_text

            self._add_section_if_valid(sections, current_heading, current_text, page_num, doc_name)
            
        return sections

    def _compute_semantic_rank(self, all_sections: list, query: str) -> list:
        """
        Computes relevance using semantic similarity with Sentence-BERT.
        """
        if not all_sections:
            return []

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        section_embeddings = self.model.encode([s['content'] for s in all_sections], convert_to_tensor=True)
        
        similarities = util.cos_sim(query_embedding, section_embeddings).flatten()

        for i, score in enumerate(similarities):
            all_sections[i]["score"] = score.item()
        
        all_sections.sort(key=lambda x: x["score"], reverse=True)

        selected_sections = []
        doc_counts = Counter()
        for section in all_sections:
            if len(selected_sections) >= self.top_k_sections:
                break
            doc_name = section["document"]
            if doc_counts[doc_name] < self.max_sections_per_document:
                selected_sections.append(section)
                doc_counts[doc_name] += 1
        return selected_sections

    def _generate_output(self, top_sections: list, input_docs: list):
        """Formats the ranked sections into a single JSON output structure as requested."""
        from pathlib import Path
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "metadata": {
                "input_documents": input_docs,
                "persona": self.persona,
                "job_to_be_done": self.job_to_be_done
            },
            "extracted_sections": [
                {
                    "document": sec["document"],
                    "section_title": sec["title"],
                    "importance_rank": i + 1,
                    "page_number": sec["page_number"]
                } for i, sec in enumerate(top_sections)
            ],
            "subsection_analysis": [
                {
                    "document": sec["document"],
                    "refined_text": sec["content"],
                    "page_number": sec["page_number"]
                } for sec in sorted(top_sections, key=lambda x: (x['document'], x['page_number']))
            ]
        }
        output_file = output_dir / "challenge1b_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    def run(self):
        """Executes the full document intelligence pipeline."""
        print("Starting Advanced Document Intelligence pipeline...")
        all_sections, input_documents = [], []
        
        if not self.documents_info:
            print(f"No documents listed in 'input.json'.")
            return

        print(f"Found {len(self.documents_info)} PDF(s) listed in input. Extracting logical sections...")
        for doc_info in self.documents_info:
            filename = doc_info['filename']
            filepath = os.path.join(self.pdf_dir, filename)

            if not os.path.exists(filepath):
                print(f"Could not find file '{filename}' in '{self.pdf_dir}' directory. Skipping.")
                continue

            try:
                all_sections.extend(self._extract_logical_sections(filepath))
                input_documents.append(filename)
            except Exception as e:
                print(f"Could not process '{filename}'. Error: {e}. Skipping file.")

        print(f"Extracted {len(all_sections)} sections. Preparing query...")
        query = f"Task for {self.persona}: {self.job_to_be_done}."

        print("Computing semantic relevance and ranking sections...")
        top_sections = self._compute_semantic_rank(all_sections, query)

        if not top_sections:
            print("No relevant sections found. Output will be generated with empty results.")
            
        print(f"Generating output for top {len(top_sections)} sections...")
        self._generate_output(top_sections, input_documents)
        print(f"Done. Output written to {self.output_dir}")


if __name__ == "__main__":
    if not os.path.exists("challenge1b_input.json"):
        print("Error: input.json not found. Please create it before running.")
    else:
        start_time = time.monotonic()
        
        intel_system = AdvancedDocumentIntel("challenge1b_input.json")
        intel_system.run()

        end_time = time.monotonic()
        duration = end_time - start_time
        print(f"Total execution time: {duration:.2f} seconds.")

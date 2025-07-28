import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import fitz  # PyMuPDF
import numpy as np
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class OptimizedPDFHeadingExtractor:
    def __init__(self):
        self.debug_mode = False
        
    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with precise formatting analysis."""
        doc = fitz.open(pdf_path)
        all_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_blocks = self._extract_page_blocks(page, page_num)
            all_blocks.extend(page_blocks)
            
        doc.close()
        return self._filter_and_clean_blocks(all_blocks)
    
    def _extract_page_blocks(self, page, page_num: int) -> List[Dict]:
        """Extract blocks from a single page with detailed analysis."""
        blocks = []
        page_rect = page.rect
        
        try:
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    if not line.get("spans"):
                        continue
                    
                    # Analyze the line's spans to get formatting
                    line_text = ""
                    font_sizes = []
                    font_flags = []
                    font_names = []
                    bboxes = []
                    
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                            
                        line_text += text + " "
                        font_sizes.append(span.get("size", 12))
                        font_flags.append(span.get("flags", 0))
                        font_names.append(span.get("font", ""))
                        bboxes.append(span.get("bbox", [0, 0, 100, 15]))
                    
                    if line_text.strip() and bboxes:
                        # Calculate line bbox
                        line_bbox = [
                            min(bbox[0] for bbox in bboxes),
                            min(bbox[1] for bbox in bboxes),
                            max(bbox[2] for bbox in bboxes),
                            max(bbox[3] for bbox in bboxes)
                        ]
                        
                        # Calculate weighted font features
                        avg_font_size = np.mean(font_sizes) if font_sizes else 12
                        dominant_flags = max(set(font_flags), key=font_flags.count) if font_flags else 0
                        
                        block_info = {
                            'text': line_text.strip(),
                            'bbox': line_bbox,
                            'page': page_num,
                            'font_size': avg_font_size,
                            'font_flags': dominant_flags,
                            'is_bold': bool(dominant_flags & 16),
                            'is_italic': bool(dominant_flags & 2),
                            'page_width': page_rect.width,
                            'page_height': page_rect.height
                        }
                        
                        # Add enhanced features
                        block_info.update(self._analyze_text_features(block_info))
                        blocks.append(block_info)
                        
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting from page {page_num}: {e}")
        
        return blocks
    
    def _analyze_text_features(self, block: Dict) -> Dict:
        """Analyze text features for heading detection."""
        text = block['text']
        
        # Basic text analysis
        word_count = len(text.split())
        char_count = len(text)
        is_caps = text.isupper() and word_count > 1
        
        # Position analysis
        page_width = block.get('page_width', 600)
        bbox = block['bbox']
        is_centered = abs((bbox[0] + bbox[2])/2 - page_width/2) < page_width * 0.15
        
        # Pattern analysis
        is_numbered = bool(re.match(r'^\d+\.?\s+', text))
        has_colon = text.strip().endswith(':')
        
        # Special patterns for different document types
        is_form_field = bool(re.search(r'^\d+\.\s*$|^[A-Z]\)\s*$', text))
        is_reference = bool(re.match(r'^\[\w+\]|\[[\d\w-]+\]', text))
        
        # Heading keywords
        heading_keywords = [
            'chapter', 'section', 'appendix', 'introduction', 'conclusion',
            'background', 'overview', 'summary', 'references', 'acknowledgements',
            'revision', 'history', 'contents', 'table', 'milestones'
        ]
        has_heading_keyword = any(keyword in text.lower() for keyword in heading_keywords)
        
        # Calculate heading score
        score = self._calculate_heading_score(block, {
            'word_count': word_count,
            'is_caps': is_caps,
            'is_centered': is_centered,
            'is_numbered': is_numbered,
            'has_colon': has_colon,
            'has_heading_keyword': has_heading_keyword,
            'is_form_field': is_form_field,
            'is_reference': is_reference
        })
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'is_caps': is_caps,
            'is_centered': is_centered,
            'is_numbered': is_numbered,
            'has_colon': has_colon,
            'has_heading_keyword': has_heading_keyword,
            'is_form_field': is_form_field,
            'is_reference': is_reference,
            'heading_score': score
        }
    
    def _calculate_heading_score(self, block: Dict, features: Dict) -> float:
        """Calculate heading likelihood score."""
        score = 0.0
        
        font_size = block['font_size']
        is_bold = block['is_bold']
        
        # Font size scoring (more aggressive)
        if font_size >= 18:
            score += 0.6
        elif font_size >= 16:
            score += 0.5
        elif font_size >= 14:
            score += 0.4
        elif font_size >= 13:
            score += 0.3
        elif font_size >= 12:
            score += 0.2
        
        # Style bonuses
        if is_bold:
            score += 0.3
        if features['is_caps'] and features['word_count'] <= 12:
            score += 0.25
        if features['is_centered']:
            score += 0.2
        
        # Pattern bonuses
        if features['has_heading_keyword']:
            score += 0.3
        if features['has_colon'] and features['word_count'] <= 8:
            score += 0.15
        if features['is_numbered'] and not features['is_form_field']:
            score += 0.1
        
        # Length considerations
        if 1 <= features['word_count'] <= 10:
            score += 0.15
        elif features['word_count'] > 20:
            score -= 0.2
        
        # Penalties
        if features['is_form_field']:
            score -= 0.4
        if features['is_reference']:
            score -= 0.3
        if block['text'].strip().isdigit():
            score -= 0.5
        if len(block['text'].strip()) < 3:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _filter_and_clean_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Filter and clean blocks to remove noise."""
        if not blocks:
            return blocks
        
        # Remove duplicates and noise
        cleaned = []
        seen_texts = set()
        
        for block in blocks:
            text = block['text'].strip()
            
            # Skip very short, numeric, or repetitive content
            if (len(text) < 3 or 
                text.isdigit() or 
                text in seen_texts or
                re.match(r'^[^\w]*$', text)):  # Only special characters
                continue
            
            seen_texts.add(text)
            cleaned.append(block)
        
        # Sort by page and position
        cleaned.sort(key=lambda x: (x['page'], x['bbox'][1], x['bbox'][0]))
        
        return cleaned
    
    def extract_title_and_outline(self, blocks: List[Dict]) -> Dict[str, Any]:
        """Extract title and outline with improved logic."""
        title = ""
        outline = []
        
        if not blocks:
            return {"title": title, "outline": outline}
        
        # Filter heading candidates
        candidates = [block for block in blocks if block['heading_score'] >= 0.3]
        
        if not candidates:
            return {"title": title, "outline": outline}
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x['heading_score'], reverse=True)
        
        # Extract title (highest scoring, early in document)
        title_candidates = [
            block for block in candidates[:5] 
            if (block['page'] <= 1 and 
                (block['is_centered'] or 
                 block['heading_score'] > 0.6 or
                 (block['font_size'] >= 16 and block['is_bold'])))
        ]
        
        if title_candidates:
            title_block = max(title_candidates, 
                            key=lambda x: (x['heading_score'], x['font_size'], x['is_centered']))
            title = title_block['text'].strip()
            
            # Remove title from candidates
            candidates = [c for c in candidates if c['text'] != title_block['text']]
        
        # Assign heading levels
        if candidates:
            # Group by font size for level assignment
            font_sizes = [block['font_size'] for block in candidates]
            unique_sizes = sorted(set(font_sizes), reverse=True)
            
            # Create level mapping based on font sizes
            level_mapping = {}
            if len(unique_sizes) >= 4:
                level_mapping = {
                    unique_sizes[0]: "H1",
                    unique_sizes[1]: "H2", 
                    unique_sizes[2]: "H3",
                    unique_sizes[3]: "H4"
                }
            elif len(unique_sizes) >= 3:
                level_mapping = {
                    unique_sizes[0]: "H1",
                    unique_sizes[1]: "H2",
                    unique_sizes[2]: "H3"
                }
            elif len(unique_sizes) >= 2:
                level_mapping = {
                    unique_sizes[0]: "H1",
                    unique_sizes[1]: "H2"
                }
            else:
                level_mapping = {unique_sizes[0]: "H1"}
            
            # Assign levels to candidates
            for block in candidates:
                font_size = block['font_size']
                
                # Find closest font size for level assignment
                closest_size = min(unique_sizes, key=lambda x: abs(x - font_size))
                base_level = level_mapping.get(closest_size, "H3")
                
                # Apply refinements
                if (block['is_caps'] and block['is_bold'] and 
                    font_size >= max(unique_sizes) * 0.95):
                    level = "H1"
                elif block['has_colon'] and base_level == "H1":
                    level = "H2"
                elif block['is_numbered'] and block['heading_score'] < 0.6:
                    level = "H3"
                else:
                    level = base_level
                
                outline.append({
                    "level": level,
                    "text": block['text'].strip(),
                    "page": block['page']
                })
        
        # Sort outline by page and position
        outline.sort(key=lambda x: (x['page'], 
                                   next((blocks[i]['bbox'][1] for i, block in enumerate(blocks) 
                                        if block['text'].strip() == x['text']), 0)))
        
        return {"title": title, "outline": outline}
    
    def extract_headings(self, pdf_path: str) -> Dict[str, Any]:
        """Main extraction method."""
        try:
            if self.debug_mode:
                print(f"Processing: {pdf_path}")
            
            blocks = self.extract_text_blocks(pdf_path)
            
            if self.debug_mode:
                print(f"Extracted {len(blocks)} blocks")
                high_score_blocks = [b for b in blocks if b['heading_score'] > 0.3]
                print(f"High-scoring blocks: {len(high_score_blocks)}")
                for block in high_score_blocks[:10]:
                    print(f"  '{block['text'][:60]}...' Score: {block['heading_score']:.3f}")
            
            result = self.extract_title_and_outline(blocks)
            
            if self.debug_mode:
                print(f"Title: '{result['title']}'")
                print(f"Outline items: {len(result['outline'])}")
            
            return result
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "", "outline": []}

def process_pdfs(input_dir: str, output_dir: str, debug: bool = False):
    """Process all PDFs and generate JSON outputs."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    extractor = OptimizedPDFHeadingExtractor()
    extractor.debug_mode = debug
    
    # Process each PDF file
    for pdf_file in input_path.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        
        try:
            result = extractor.extract_headings(str(pdf_file))
            
            # Save result
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved: {output_file.name}")
            print(f"  Title: '{result['title']}'")
            print(f"  Outline: {len(result['outline'])} items")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")
            # Save empty result for failed files
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"title": "", "outline": []}, f, indent=2)

if __name__ == "__main__":
    input_directory = "app/input"
    output_directory = "app/output"
    
    # Create directories
    os.makedirs(input_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    
    # Process PDFs with debug output
    process_pdfs(input_directory, output_directory, debug=True)
    print("Processing complete!")

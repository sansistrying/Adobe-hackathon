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

# Optional: Add sentence-transformers for better semantic analysis
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

class GeneralizedPDFHeadingExtractor:
    def __init__(self, use_embeddings: bool = True):
        self.debug_mode = False
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        
        # Dynamic thresholds that adapt to each document
        self.adaptive_thresholds = {}
        
        # Initialize lightweight embedding model if available
        if self.use_embeddings:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_model.max_seq_length = 128
                if self.debug_mode:
                    print("✓ Loaded embedding model")
            except Exception as e:
                if self.debug_mode:
                    print(f"Warning: Could not load embedding model: {e}")
                self.use_embeddings = False
    
    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with adaptive formatting analysis."""
        doc = fitz.open(pdf_path)
        all_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_blocks = self._extract_page_blocks(page, page_num)
            all_blocks.extend(page_blocks)
            
        doc.close()
        
        # Calculate document-specific statistics for adaptive processing
        self._calculate_document_statistics(all_blocks)
        
        # Apply adaptive filtering and cleaning
        return self._adaptive_filter_and_clean(all_blocks)
    
    def _extract_page_blocks(self, page, page_num: int) -> List[Dict]:
        """Extract blocks with multiple fallback methods."""
        blocks = []
        page_rect = page.rect
        
        # Method 1: Dictionary-based extraction (most detailed)
        try:
            text_dict = page.get_text("dict")
            blocks = self._extract_from_dict(text_dict, page_num, page_rect)
        except Exception as e:
            if self.debug_mode:
                print(f"Dict extraction failed on page {page_num}: {e}")
        
        # Method 2: Block-based extraction (fallback)
        if not blocks:
            try:
                text_blocks = page.get_text("blocks")
                blocks = self._extract_from_blocks(text_blocks, page_num, page_rect)
            except Exception as e:
                if self.debug_mode:
                    print(f"Block extraction failed on page {page_num}: {e}")
        
        # Method 3: Simple text extraction (last resort)
        if not blocks:
            try:
                simple_text = page.get_text()
                blocks = self._extract_from_simple_text(simple_text, page_num, page_rect)
            except Exception as e:
                if self.debug_mode:
                    print(f"Simple extraction failed on page {page_num}: {e}")
        
        return blocks
    
    def _extract_from_dict(self, text_dict: Dict, page_num: int, page_rect) -> List[Dict]:
        """Extract from dictionary format with comprehensive error handling."""
        blocks = []
        
        for block in text_dict.get("blocks", []):
            if not isinstance(block, dict) or "lines" not in block:
                continue
                
            for line in block.get("lines", []):
                if not isinstance(line, dict) or "spans" not in line:
                    continue
                
                line_text = ""
                font_info = []
                bboxes = []
                
                for span in line.get("spans", []):
                    if not isinstance(span, dict):
                        continue
                    
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    line_text += text + " "
                    font_info.append({
                        'size': span.get('size', 12),
                        'flags': span.get('flags', 0),
                        'font': span.get('font', 'default'),
                        'text_length': len(text)
                    })
                    bboxes.append(span.get("bbox", [0, 0, 100, 15]))
                
                if line_text.strip() and bboxes:
                    block_info = self._create_block_info(
                        line_text.strip(), bboxes, font_info, page_num, page_rect
                    )
                    blocks.append(block_info)
        
        return blocks
    
    def _extract_from_blocks(self, text_blocks: List, page_num: int, page_rect) -> List[Dict]:
        """Fallback extraction from blocks format."""
        blocks = []
        
        for block in text_blocks:
            if len(block) >= 7 and block[4].strip():
                block_info = {
                    'text': block[4].strip(),
                    'bbox': list(block[:4]),
                    'page': page_num,
                    'font_size': 12,  # Default since not available
                    'font_flags': 0,
                    'is_bold': False,
                    'is_italic': False,
                    'page_width': page_rect.width,
                    'page_height': page_rect.height,
                    'extraction_method': 'blocks'
                }
                block_info.update(self._analyze_text_features(block_info))
                blocks.append(block_info)
        
        return blocks
    
    def _extract_from_simple_text(self, simple_text: str, page_num: int, page_rect) -> List[Dict]:
        """Last resort extraction from simple text."""
        blocks = []
        
        if simple_text.strip():
            lines = [line.strip() for line in simple_text.split('\n') if line.strip()]
            
            for i, line in enumerate(lines):
                block_info = {
                    'text': line,
                    'bbox': [0, i*15, page_rect.width, (i+1)*15],
                    'page': page_num,
                    'font_size': 12,
                    'font_flags': 0,
                    'is_bold': False,
                    'is_italic': False,
                    'page_width': page_rect.width,
                    'page_height': page_rect.height,
                    'extraction_method': 'simple'
                }
                block_info.update(self._analyze_text_features(block_info))
                blocks.append(block_info)
        
        return blocks
    
    def _create_block_info(self, text: str, bboxes: List, font_info: List[Dict], 
                          page_num: int, page_rect) -> Dict:
        """Create comprehensive block information."""
        # Calculate bbox
        bbox = [
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes)
        ]
        
        # Calculate weighted font features
        if font_info:
            total_chars = sum(f['text_length'] for f in font_info)
            if total_chars > 0:
                weighted_size = sum(f['size'] * f['text_length'] for f in font_info) / total_chars
                weighted_flags = sum(f['flags'] * f['text_length'] for f in font_info) / total_chars
            else:
                weighted_size = np.mean([f['size'] for f in font_info])
                weighted_flags = np.mean([f['flags'] for f in font_info])
        else:
            weighted_size = 12
            weighted_flags = 0
        
        block_info = {
            'text': text,
            'bbox': bbox,
            'page': page_num,
            'font_size': weighted_size,
            'font_flags': weighted_flags,
            'is_bold': bool(int(weighted_flags) & 16),
            'is_italic': bool(int(weighted_flags) & 2),
            'page_width': page_rect.width,
            'page_height': page_rect.height,
            'extraction_method': 'dict'
        }
        
        # Add text analysis features
        block_info.update(self._analyze_text_features(block_info))
        return block_info
    
    def _analyze_text_features(self, block: Dict) -> Dict:
        """Analyze text features without hardcoded patterns."""
        text = block['text']
        
        # Basic measurements
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Position analysis
        page_width = block.get('page_width', 600)
        bbox = block['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        is_centered = abs(center_x - page_width/2) < page_width * 0.15
        
        # Pattern detection (generalized)
        has_colon_end = text.strip().endswith(':')
        starts_with_number = bool(re.match(r'^\d+', text.strip()))
        has_special_chars = bool(re.search(r'[•▪▫►▼◄▲→←↑↓]', text))
        is_all_caps = text.isupper() and word_count > 1
        
        # Language-agnostic patterns
        has_parentheses = '(' in text and ')' in text
        has_brackets = '[' in text and ']' in text
        ends_with_period = text.strip().endswith('.')
        
        # Calculate initial heading score
        score = self._calculate_adaptive_heading_score(block, {
            'word_count': word_count,
            'char_count': char_count,
            'is_centered': is_centered,
            'has_colon_end': has_colon_end,
            'starts_with_number': starts_with_number,
            'has_special_chars': has_special_chars,
            'is_all_caps': is_all_caps,
            'has_parentheses': has_parentheses,
            'has_brackets': has_brackets,
            'ends_with_period': ends_with_period
        })
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'is_centered': is_centered,
            'has_colon_end': has_colon_end,
            'starts_with_number': starts_with_number,
            'has_special_chars': has_special_chars,
            'is_all_caps': is_all_caps,
            'has_parentheses': has_parentheses,
            'has_brackets': has_brackets,
            'ends_with_period': ends_with_period,
            'heading_score': score
        }
    
    def _calculate_document_statistics(self, blocks: List[Dict]):
        """Calculate document-specific statistics for adaptive processing."""
        if not blocks:
            return
        
        # Font size statistics
        font_sizes = [b['font_size'] for b in blocks]
        self.adaptive_thresholds['font_percentiles'] = np.percentile(
            font_sizes, [10, 25, 50, 75, 90, 95]
        ) if font_sizes else [10, 11, 12, 13, 14, 16]
        
        # Text length statistics
        word_counts = [b['word_count'] for b in blocks]
        self.adaptive_thresholds['length_percentiles'] = np.percentile(
            word_counts, [10, 25, 50, 75, 90]
        ) if word_counts else [1, 3, 8, 15, 25]
        
        # Document characteristics
        total_blocks = len(blocks)
        bold_blocks = sum(1 for b in blocks if b['is_bold'])
        caps_blocks = sum(1 for b in blocks if b.get('is_all_caps', False))
        centered_blocks = sum(1 for b in blocks if b.get('is_centered', False))
        
        self.adaptive_thresholds['bold_ratio'] = bold_blocks / total_blocks if total_blocks > 0 else 0
        self.adaptive_thresholds['caps_ratio'] = caps_blocks / total_blocks if total_blocks > 0 else 0
        self.adaptive_thresholds['centered_ratio'] = centered_blocks / total_blocks if total_blocks > 0 else 0
        
        if self.debug_mode:
            print(f"Document stats - Font percentiles: {self.adaptive_thresholds['font_percentiles']}")
            print(f"Bold ratio: {self.adaptive_thresholds['bold_ratio']:.3f}")
    
    def _calculate_adaptive_heading_score(self, block: Dict, features: Dict) -> float:
        """Calculate heading score using adaptive thresholds."""
        score = 0.0
        
        font_size = block['font_size']
        font_percentiles = self.adaptive_thresholds.get('font_percentiles', [10, 11, 12, 13, 14, 16])
        
        # Adaptive font size scoring
        if font_size >= font_percentiles[5]:  # 95th percentile
            score += 0.6
        elif font_size >= font_percentiles[4]:  # 90th percentile
            score += 0.5
        elif font_size >= font_percentiles[3]:  # 75th percentile
            score += 0.4
        elif font_size >= font_percentiles[2]:  # 50th percentile
            score += 0.3
        elif font_size >= font_percentiles[1]:  # 25th percentile
            score += 0.2
        
        # Style contributions (adaptive to document)
        bold_ratio = self.adaptive_thresholds.get('bold_ratio', 0.1)
        if block['is_bold']:
            # If many blocks are bold, reduce the bonus
            score += max(0.1, 0.4 - bold_ratio)
        
        caps_ratio = self.adaptive_thresholds.get('caps_ratio', 0.1)
        if features['is_all_caps']:
            score += max(0.1, 0.3 - caps_ratio)
        
        centered_ratio = self.adaptive_thresholds.get('centered_ratio', 0.1)
        if features['is_centered']:
            score += max(0.05, 0.25 - centered_ratio)
        
        # Length-based scoring (adaptive)
        length_percentiles = self.adaptive_thresholds.get('length_percentiles', [1, 3, 8, 15, 25])
        word_count = features['word_count']
        
        if word_count <= length_percentiles[1]:  # Very short
            score += 0.2
        elif word_count <= length_percentiles[2]:  # Short
            score += 0.15
        elif word_count > length_percentiles[4]:  # Very long
            score -= 0.15
        
        # Pattern bonuses (universal)
        if features['has_colon_end'] and word_count <= 10:
            score += 0.15
        if features['starts_with_number'] and not features['ends_with_period']:
            score += 0.1
        if features['has_special_chars']:
            score += 0.05
        
        # Penalties for unlikely heading patterns
        if features['has_brackets'] and word_count < 5:
            score -= 0.2
        if text := block['text'].strip():
            if text.isdigit() or len(text) < 3:
                score -= 0.4
            if features['ends_with_period'] and word_count > 15:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _adaptive_filter_and_clean(self, blocks: List[Dict]) -> List[Dict]:
        """Adaptive filtering based on document characteristics."""
        if not blocks:
            return blocks
        
        # Remove duplicates
        seen = set()
        unique_blocks = []
        
        for block in blocks:
            # Create a key based on text and approximate position
            key = (
                block['text'][:50],  # First 50 chars
                block['page'],
                round(block['bbox'][1], -1)  # Round Y position to nearest 10
            )
            
            if key not in seen:
                seen.add(key)
                unique_blocks.append(block)
        
        # Adaptive noise filtering
        cleaned = []
        
        for block in unique_blocks:
            text = block['text'].strip()
            
            # Universal noise patterns
            if (len(text) < 2 or 
                re.match(r'^[^\w\s]*$', text) or  # Only punctuation
                re.match(r'^\d+$', text)):  # Only numbers
                continue
            
            # Skip very common, non-heading patterns
            if (re.match(r'^page \d+$', text.lower()) or
                re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', text) or  # Dates
                re.match(r'^[a-zA-Z]?[\d\s-]+$', text)):  # Phone numbers, IDs
                continue
            
            cleaned.append(block)
        
        # Sort by page and position
        cleaned.sort(key=lambda x: (x['page'], x['bbox'][1], x['bbox'][0]))
        
        return cleaned
    
    def detect_semantic_boundaries_enhanced(self, blocks: List[Dict]) -> List[float]:
        """Enhanced semantic boundary detection."""
        if len(blocks) < 2:
            return [0.0] * len(blocks)
        
        discontinuities = [0.0]
        
        # Use embeddings for semantic analysis if available and reasonable size
        if self.use_embeddings and len(blocks) <= 100:
            try:
                texts = [block['text'] for block in blocks]
                embeddings = self.embedding_model.encode(texts, batch_size=16, show_progress_bar=False)
                
                for i in range(1, len(blocks)):
                    similarity = cosine_similarity(
                        embeddings[i-1:i], 
                        embeddings[i:i+1]
                    )[0][0]
                    
                    discontinuity = 1.0 - similarity
                    
                    # Add visual and positional factors
                    discontinuity = self._enhance_discontinuity(
                        discontinuity, blocks[i-1], blocks[i]
                    )
                    
                    discontinuities.append(discontinuity)
                
                return discontinuities
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Embedding analysis failed: {e}")
        
        # Fallback: keyword-based analysis
        for i in range(1, len(blocks)):
            discontinuity = self._calculate_text_discontinuity(
                blocks[i-1]['text'], blocks[i]['text']
            )
            
            discontinuity = self._enhance_discontinuity(
                discontinuity, blocks[i-1], blocks[i]
            )
            
            discontinuities.append(discontinuity)
        
        return discontinuities
    
    def _calculate_text_discontinuity(self, prev_text: str, curr_text: str) -> float:
        """Calculate text-based discontinuity."""
        # Extract meaningful words (length > 2)
        prev_words = set(word.lower() for word in prev_text.split() if len(word) > 2)
        curr_words = set(word.lower() for word in curr_text.split() if len(word) > 2)
        
        if not prev_words or not curr_words:
            return 0.7
        
        # Jaccard similarity
        intersection = len(prev_words & curr_words)
        union = len(prev_words | curr_words)
        similarity = intersection / union if union > 0 else 0.0
        
        return 1.0 - similarity
    
    def _enhance_discontinuity(self, base_discontinuity: float, prev_block: Dict, curr_block: Dict) -> float:
        """Enhance discontinuity with visual and positional factors."""
        enhanced = base_discontinuity
        
        # Visual gap
        gap = curr_block['bbox'][1] - prev_block['bbox'][3]
        if gap > 30:
            enhanced = min(1.0, enhanced + 0.3)
        elif gap > 15:
            enhanced = min(1.0, enhanced + 0.2)
        
        # Page break
        if curr_block['page'] != prev_block['page']:
            enhanced = min(1.0, enhanced + 0.4)
        
        # Style change
        if (abs(curr_block['font_size'] - prev_block['font_size']) > 2 or
            curr_block['is_bold'] != prev_block['is_bold']):
            enhanced = min(1.0, enhanced + 0.2)
        
        return enhanced
    
    def extract_title_and_outline(self, blocks: List[Dict]) -> Dict[str, Any]:
        """Extract title and outline with fully adaptive logic."""
        title = ""
        outline = []
        
        if not blocks:
            return {"title": title, "outline": outline}
        
        # Apply semantic analysis
        discontinuities = self.detect_semantic_boundaries_enhanced(blocks)
        
        # Boost scores based on semantic discontinuities
        for i, block in enumerate(blocks):
            if i < len(discontinuities) and discontinuities[i] > 0.6:
                block['heading_score'] = min(1.0, block['heading_score'] + 0.15)
        
        # Adaptive candidate filtering
        score_threshold = self._calculate_adaptive_threshold(blocks)
        candidates = [block for block in blocks if block['heading_score'] >= score_threshold]
        
        if not candidates:
            return {"title": title, "outline": outline}
        
        # Extract title using adaptive criteria
        title_block = self._find_title_adaptively(candidates)
        if title_block:
            title = title_block['text'].strip()
            candidates = [c for c in candidates if c['text'] != title_block['text']]
        
        # Assign levels adaptively
        if candidates:
            outline = self._assign_levels_adaptively(candidates, blocks)
        
        # Sort by appearance order
        outline.sort(key=lambda x: (x['page'], 
                                   next((blocks[i]['bbox'][1] for i, block in enumerate(blocks) 
                                        if block['text'].strip() == x['text']), 0)))
        
        return {"title": title, "outline": outline}
    
    def _calculate_adaptive_threshold(self, blocks: List[Dict]) -> float:
        """Calculate adaptive threshold for heading candidates."""
        scores = [block['heading_score'] for block in blocks]
        
        if not scores:
            return 0.3
        
        # Use score distribution to determine threshold
        score_percentiles = np.percentile(scores, [70, 80, 90])
        
        # If document has many high-scoring blocks, raise threshold
        high_score_ratio = sum(1 for s in scores if s > 0.5) / len(scores)
        
        if high_score_ratio > 0.2:  # Many potential headings
            return max(0.4, score_percentiles[1])  # 80th percentile
        elif high_score_ratio > 0.1:
            return max(0.35, score_percentiles[0])  # 70th percentile
        else:
            return 0.3  # Default threshold
    
    def _find_title_adaptively(self, candidates: List[Dict]) -> Optional[Dict]:
        """Find title using adaptive criteria."""
        if not candidates:
            return None
        
        # Look for title in first few pages
        early_candidates = [c for c in candidates if c['page'] <= 2]
        if not early_candidates:
            early_candidates = candidates[:5]
        
        # Score title candidates
        title_scores = []
        for candidate in early_candidates:
            score = candidate['heading_score']
            
            # Bonuses for title characteristics
            if candidate['is_centered']:
                score += 0.3
            if candidate['font_size'] >= self.adaptive_thresholds.get('font_percentiles', [0,0,0,0,0,16])[4]:
                score += 0.2
            if candidate['page'] == 0:
                score += 0.1
            if 1 <= candidate['word_count'] <= 12:
                score += 0.1
            
            title_scores.append((candidate, score))
        
        # Return highest scoring title candidate
        if title_scores:
            return max(title_scores, key=lambda x: x[1])[0]
        
        return None
    
    def _assign_levels_adaptively(self, candidates: List[Dict], all_blocks: List[Dict]) -> List[Dict]:
        """Assign heading levels adaptively based on document structure."""
        outline = []
        
        # Group candidates by font size
        font_sizes = [c['font_size'] for c in candidates]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        # Create flexible level mapping
        level_names = ["H1", "H2", "H3", "H4"]
        level_mapping = {}
        
        for i, size in enumerate(unique_sizes[:4]):
            level_mapping[size] = level_names[i]
        
        # If only one size, assign based on other criteria
        if len(unique_sizes) == 1:
            level_mapping[unique_sizes[0]] = "H1"
        
        # Assign levels to candidates
        for candidate in candidates:
            font_size = candidate['font_size']
            
            # Find closest font size
            closest_size = min(unique_sizes, key=lambda x: abs(x - font_size))
            base_level = level_mapping.get(closest_size, "H3")
            
            # Apply refinements based on content and style
            final_level = self._refine_level(candidate, base_level, unique_sizes)
            
            outline.append({
                "level": final_level,
                "text": candidate['text'].strip(),
                "page": candidate['page']
            })
        
        return outline
    
    def _refine_level(self, candidate: Dict, base_level: str, font_sizes: List[float]) -> str:
        """Refine heading level based on content and style."""
        # If highly formatted and large font, likely H1
        if (candidate['is_bold'] and candidate['is_all_caps'] and 
            candidate['font_size'] >= max(font_sizes) * 0.95):
            return "H1"
        
        # If ends with colon, likely section header
        if candidate['has_colon_end'] and base_level == "H1":
            return "H2"
        
        # If starts with number and not too high score, likely subsection
        if (candidate['starts_with_number'] and 
            candidate['heading_score'] < 0.7 and 
            base_level in ["H1", "H2"]):
            return "H3"
        
        return base_level
    
    def extract_headings(self, pdf_path: str) -> Dict[str, Any]:
        """Main extraction method - fully generalized."""
        try:
            if self.debug_mode:
                print(f"Processing: {pdf_path}")
            
            blocks = self.extract_text_blocks(pdf_path)
            
            if self.debug_mode:
                print(f"Extracted {len(blocks)} blocks")
                threshold = self._calculate_adaptive_threshold(blocks) if blocks else 0.3
                high_score_blocks = [b for b in blocks if b['heading_score'] >= threshold]
                print(f"Candidates (threshold={threshold:.3f}): {len(high_score_blocks)}")
            
            result = self.extract_title_and_outline(blocks)
            
            if self.debug_mode:
                print(f"Title: '{result['title']}'")
                print(f"Outline items: {len(result['outline'])}")
            
            return result
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "", "outline": []}

def process_pdfs(input_dir: str, output_dir: str, debug: bool = False, use_embeddings: bool = True):
    """Process all PDFs with fully generalized extraction."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    extractor = GeneralizedPDFHeadingExtractor(use_embeddings=use_embeddings)
    extractor.debug_mode = debug
    
    for pdf_file in input_path.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        
        try:
            result = extractor.extract_headings(str(pdf_file))
            
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved: {output_file.name}")
            print(f"  Title: '{result['title']}'")
            print(f"  Outline: {len(result['outline'])} items")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"title": "", "outline": []}, f, indent=2)

if __name__ == "__main__":
    input_directory = "app/input"
    output_directory = "app/output"
    
    os.makedirs(input_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    
    process_pdfs(input_directory, output_directory, debug=True, use_embeddings=True)
    print("Processing complete!")

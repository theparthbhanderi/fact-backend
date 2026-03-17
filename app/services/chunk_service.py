import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def chunk_text(text: str, source_metadata: Dict, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict]:
    """
    Splits long Markdown text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    Attaches the source_metadata to each chunk.
    """
    if not text:
        return []
        
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        raw_chunks = text_splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            # Duplicate the metadata and inject the chunk text
            chunk_dict = source_metadata.copy()
            chunk_dict["text"] = chunk_text
            chunk_dict["chunk_id"] = i
            chunks.append(chunk_dict)
            
        logger.info(f"🔪 Chunked article '{source_metadata.get('title', 'Unknown')}' into {len(chunks)} segments.")
        return chunks
        
    except ImportError:
        logger.warning("langchain not found. Falling back to simple overlap chunking.")
        return _fallback_chunking(text, source_metadata, chunk_size, chunk_overlap)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        return [source_metadata.copy().update({"text": text})]

def _fallback_chunking(text: str, metadata: Dict, chunk_size: int, overlap: int) -> List[Dict]:
    """Simple overlapping string slice chunking (characters, not exactly tokens)."""
    chunks = []
    start = 0
    text_len = len(text)
    
    # Very crude approximation: 1 token ~= 4 chars
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    
    i = 0
    while start < text_len:
        end = start + char_size
        chunk_str = text[start:end]
        
        chunk_dict = metadata.copy()
        chunk_dict["text"] = chunk_str
        chunk_dict["chunk_id"] = i
        chunks.append(chunk_dict)
        
        start += (char_size - char_overlap)
        i += 1
        
    return chunks

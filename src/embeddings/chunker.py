# src/embeddings/chunker.py
import re
from dataclasses import dataclass

from loguru import logger

from src.services.token_counter import count_tokens, truncate_text_to_tokens


@dataclass
class Chunk:
    """
    A text chunk ready for embedding and storage.

    Attributes:
        text: The chunk content.
        index: Position in the original document (0-based).
        token_count: Approximate token count for this chunk.
        metadata: Arbitrary metadata (source file, page number, section, etc.).
    """

    text: str
    index: int
    token_count: int
    metadata: dict


def chunk_fixed_size(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text into fixed-size token chunks with overlap.

    Overlap prevents information loss at chunk boundaries — a sentence
    spanning the boundary of two chunks will be fully present in at
    least one of them.

    Args:
        text: Raw text to split.
        chunk_size: Target size in tokens for each chunk.
        overlap: Number of tokens to repeat between consecutive chunks.
        metadata: Document-level metadata attached to every chunk.

    Returns:
        List of Chunk objects ready for embedding.
    """
    if not text.strip():
        return []

    meta = metadata or {}
    words = text.split()
    chunks = []
    index = 0

    # Approximate word-level splitting (words ~ tokens for estimation)
    # We refine with actual token counting after splitting

    start_word = 0
    while start_word < len(words):
        # Build candidate chunk by adding words until token limit reached
        candidate_words = words[start_word:]
        candidate_text = " ".join(candidate_words)

        # Truncate to exact token limit
        chunk_text = truncate_text_to_tokens(candidate_text, chunk_size)
        tokens = count_tokens(chunk_text)

        if not chunk_text.strip():
            break

        chunks.append(
            Chunk(
                text=chunk_text.strip(),
                index=index,
                token_count=tokens,
                metadata={**meta, "chunk_strategy": "fixed_size"},
            )
        )

        # Advance by step (chunk_size - overlap) words
        # This is an approximation — actual overlap in tokens may vary
        words_in_chunk = len(chunk_text.split())
        advance = max(1, words_in_chunk - overlap // 2)
        start_word += advance
        index += 1

    logger.debug(
        "Fixed-size chunking | input_chars={} | chunks={} | chunk_size={} | overlap={}",
        len(text),
        len(chunks),
        chunk_size,
        overlap,
    )

    return chunks


def chunk_recursive(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text recursively by natural separators (paragraphs, lines, sentences).

    Strategy: try to split by paragraphs first (double newline), then single
    newlines, then sentences, then words. Merges small splits and further
    splits large ones.

    This is the most commonly used strategy in production RAG systems
    (equivalent to LangChain's RecursiveCharacterTextSplitter).

    Args:
        text: Raw text to split.
        chunk_size: Target chunk size in tokens.
        overlap: Overlap between consecutive chunks in tokens.
        metadata: Document-level metadata.

    Returns:
        List of Chunk objects.
    """
    meta = metadata or {}

    # Separator hierarchy : paragraphs > lines > sentences > words
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, sep_idx: int) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if count_tokens(text) <= chunk_size:
            return [text] if text.strip() else []

        if sep_idx >= len(separators):
            # Last resort: truncate
            return [truncate_text_to_tokens(text, chunk_size)]

        sep = separators[sep_idx]
        parts = text.split(sep)

        result = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()

            if count_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # Part itself may be too large -> recurse with next separator
                if count_tokens(part) > chunk_size:
                    result.extend(_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part.strip()

        if current:
            result.append(current)

        return result

    raw_chunks = _split(text, sep_idx=0)

    # Add overlap: prepend end of previous chunk to current chunk
    chunks_with_overlap = []
    for i, chunk_text in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            # Take last `overlap` tokens from previous chunk
            prev_tokens = count_tokens(raw_chunks[i - 1])
            if prev_tokens > overlap:
                # Estimate character count for overlap tokens
                overlap_chars = int(len(raw_chunks[i - 1]) * overlap / prev_tokens)
                prefix = raw_chunks[i - 1][-overlap_chars:]
                chunk_text = prefix.strip() + " " + chunk_text.strip()
                chunk_text = truncate_text_to_tokens(chunk_text, chunk_size)

        chunks_with_overlap.append(chunk_text.strip())

    result = []
    for i, text_chunk in enumerate(chunks_with_overlap):
        if text_chunk:
            result.append(
                Chunk(
                    text=text_chunk,
                    index=i,
                    token_count=count_tokens(text_chunk),
                    metadata={**meta, "chunk_strategy": "recursive"},
                )
            )

    logger.debug(
        "Recursive chunking | input_chars={} | chunks={} | chunk_size={} | overlap={}",
        len(text),
        len(result),
        chunk_size,
        overlap,
    )

    return result


def chunk_by_article(
    text: str,
    metadata: dict | None = None,
    max_chunk_size: int = 512,
) -> list[Chunk]:
    """
    Split regulatory text by article/section markers.

    Optimized for GDPR, CNIL recommendations, and ANSSI guides where
    each article is a natural semantic unit. Articles that exceed
    max_chunk_size are further split with chunk_recursive().

    Detected patterns:
      - "Article 5", "Art. 5", "ARTICLE 5"
      - "Section 3", "Chapitre II"
      - "3.1", "3.1.2" (numbered sections)

    Args:
        text: Raw regulatory document text.
        metadata: Document-level metadata.
        max_chunk_size: Maximum tokens per chunk (oversized articles are split).

    Returns:
        List of Chunk objects, one per article/section.
    """
    meta = metadata or {}

    # Regex patterns for regulatory document structure
    article_pattern = re.compile(
        r"(?:^|\n)(?:"
        r"(?:Article|Art\.?|ARTICLE)\s+\d+[a-z]?"  # Article 5, Art. 5
        r"|(?:Section|Chapitre|SECTION|CHAPITRE)\s+\w+"  # Section 3
        r"|\d+\.\d+(?:\.\d+)?"  # 3.1, 3.1.2
        r")",
        re.MULTILINE,
    )

    # Find split points
    matches = list(article_pattern.finditer(text))

    if not matches:
        logger.debug("No article markers found, falling back to recursive chunking")
        return chunk_recursive(text, metadata=metadata, chunk_size=max_chunk_size)

    # Extract article sections
    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append(section_text)

    # Handle text before first article marker
    preamble = text[: matches[0].start()].strip()
    if preamble and count_tokens(preamble) > 50:  # ignore very short preambles
        sections.insert(0, preamble)

    # Split oversized sections further
    chunks = []
    chunk_index = 0
    for section in sections:
        if count_tokens(section) <= max_chunk_size:
            chunks.append(
                Chunk(
                    text=section,
                    index=chunk_index,
                    token_count=count_tokens(section),
                    metadata={**meta, "chunk_strategy": "by_article"},
                )
            )
            chunk_index += 1
        else:
            # Article too long: split recursively but keep article metadata
            sub_chunks = chunk_recursive(
                section,
                chunk_size=max_chunk_size,
                metadata={**meta, "chunk_strategy": "by_article_split"},
            )
            for sub in sub_chunks:
                sub.index = chunk_index
                chunks.append(sub)
                chunk_index += 1

    logger.info(
        "Article chunking | sections={} | chunks={} | input_chars={}",
        len(sections),
        len(chunks),
        len(text),
    )

    return chunks


def select_chunking_strategy(
    text: str,
    document_type: str = "generic",
    chunk_size: int = 512,
    overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Select the best chunking strategy based on document type.

    Args:
        text: Document text.
        document_type: "regulatory" (RGPD/CNIL/ANSSI), "generic", or "structured".
        chunk_size: Target chunk size in tokens.
        overlap: Overlap between chunks.
        metadata: Document metadata.

    Returns:
        List of chunks using the appropriate strategy.
    """
    if document_type == "regulatory":
        # Try article-based first — ideal for RGPD, CNIL, ANSSI
        chunks = chunk_by_article(text, metadata=metadata, max_chunk_size=chunk_size)
        if len(chunks) > 1:
            return chunks
        # Fallback if no article markers detected
        return chunk_recursive(text, chunk_size, overlap, metadata)

    elif document_type == "structured":
        return chunk_recursive(text, chunk_size, overlap, metadata)

    else:
        return chunk_fixed_size(text, chunk_size, overlap, metadata)

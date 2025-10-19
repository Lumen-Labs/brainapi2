"""
File: /text_chunker.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 8:52:43 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import gc
import threading
from typing import List
import chunker_cpp


# Prevent multiple threads from accessing the chunker at the same time.
_chunker_lock = threading.Lock()


def chunk_text(
    text: str,
    max_chunk_size: int = 2000,
    min_chunk_size: int = 500,
    min_coherence_threshold: float = 0.3,
) -> List[str]:
    """
    Chunk text into smaller chunks.

    Args:
        text: The text to chunk.
        max_chunk_size: The maximum size of each chunk.
        min_chunk_size: The minimum size of each chunk.
        min_coherence_threshold: The minimum coherence threshold for the chunks.

    Returns:
        A list of chunks.

    Raises:
        Exception: If the chunking fails.
    """

    try:
        if not text or len(text.strip()) == 0:
            return [text] if text else []

        with _chunker_lock:
            try:
                result = chunker_cpp.chunk_text_semantically(
                    text, max_chunk_size, min_chunk_size, min_coherence_threshold
                )

                # Force garbage collection after C++ call
                gc.collect()

                if not result or not isinstance(result, list):
                    return [text]

                return result

            except Exception as cpp_error:
                print(f"C++ chunker error: {cpp_error}")
                raise

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Chunker failed: {e}, falling back to simple chunking")
        chunk_size = min(max_chunk_size, 2000)
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]

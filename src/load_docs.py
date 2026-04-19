from dataclasses import dataclass
from pathlib import Path

# HIGH LEVEL: turns source docs (.rst) into small text chunks with metadata

# structured dataclass
@dataclass(frozen=True) # immutable after creation
class DocChunk:
    filename: str
    startline: int
    endline: int
    text: str

# line chunker function, input: (document split into lines, lines per chunk, and how many lines neighboring chunks share), output list which is chunk starting line, ending line, and chunk text
def _chunk_lines(lines: list[str], chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:

    # note: I used overlap as a parameter over step size as how much context adjacent chunks share is more intuitive

    # chunk validation
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[tuple[int, int, str]] = [] # list of chunks to be outputted

    step = chunk_size - overlap # how far the chunk moves each iteration



    index = 0
    total = len(lines)

    while index < total:
        start_idx = index
        end_idx = min(index + chunk_size, total)
        block = lines[start_idx:end_idx]
        text = "\n".join(block).strip()

        if text: # if empty, append to chunk list
            # convert 0-based indexes to 1-based line numbers.
            chunks.append((start_idx + 1, end_idx, text))

        index += step

    return chunks # return list of chunks, each item is (start line, end line, chunk text)


# apply chunk_lines() function to every .rst doc
def load_corpus_chunks(
    docs_dir: Path,
    chunk_size: int = 40, # knob
    overlap: int = 10, # knob
) -> list[DocChunk]:
    
    if not docs_dir.exists(): # validate folder
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    all_chunks: list[DocChunk] = [] # initiliaze list to be outputted

    for rst_path in sorted(docs_dir.glob("*.rst")): # for each .rst file, split into lines, call chunk_line() function and append to all_chunks list
        with rst_path.open("r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        filename = rst_path.name

        for startline, endline, text in _chunk_lines(lines, chunk_size, overlap):
            all_chunks.append(
                DocChunk(
                    filename=filename,
                    startline=startline,
                    endline=endline,
                    text=text,
                )
            )

    return all_chunks # list of all chunks from all .rst files

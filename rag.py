# import argparse
# import os
# import re
# import json
# from typing import List, Tuple
#
# # LangChain (>=0.2)
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.documents import Document
#
# # -------- Sentence splitter (VN/EN friendly) --------
# SENTENCE_SPLIT_REGEX = r"(?<=[\.\?\!…])\s+|\n+"
#
# # Default multilingual model (better for VN+EN)
# DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#
# def get_embedder(model_name: str) -> HuggingFaceEmbeddings:
#     """
#     Wrap sentence-transformers in LangChain with cosine-friendly normalization.
#     """
#     return HuggingFaceEmbeddings(
#         model_name=model_name,
#         encode_kwargs={"normalize_embeddings": True}  # cosine = dot product
#     )
#
# def read_text(path: str) -> str:
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read()
#
# def split_into_sentences(text: str) -> List[str]:
#     text = re.sub(r"\s+", " ", text).strip()
#     if not text:
#         return []
#     raw = re.split(SENTENCE_SPLIT_REGEX, text)
#     sentences = [s.strip() for s in raw if len(s.strip()) >= 2]
#     return sentences
#
# def build_docs(sentences: List[str], source_path: str) -> List[Document]:
#     docs = []
#     for idx, s in enumerate(sentences):
#         meta = {"source": os.path.basename(source_path), "sentence_id": idx}
#         docs.append(Document(page_content=s, metadata=meta))
#     return docs
#
# def index_corpus(input_txt: str, index_dir: str, model_name: str):
#     text = read_text(input_txt)
#     sentences = split_into_sentences(text)
#     if not sentences:
#         raise ValueError("Không tìm thấy câu nào để lập chỉ mục (file rỗng?).")
#
#     print(f"[Index] Số câu: {len(sentences)}")
#     docs = build_docs(sentences, input_txt)
#
#     emb = get_embedder(model_name)
#     vs = FAISS.from_documents(docs, emb)
#
#     os.makedirs(index_dir, exist_ok=True)
#     vs.save_local(index_dir)
#     with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
#         json.dump(
#             {
#                 "source_file": os.path.abspath(input_txt),
#                 "num_sentences": len(docs),
#                 "model": model_name,
#             },
#             f,
#             ensure_ascii=False,
#             indent=2,
#         )
#     print(f"[Index] Đã lưu FAISS vào: {index_dir}")
#     print(f"[Index] Model embeddings: {model_name}")
#
# def load_index(index_dir: str, fallback_model: str) -> Tuple[FAISS, str]:
#     """
#     Load FAISS with the SAME embedding model used at index time.
#     If meta.json missing, fall back to provided model.
#     """
#     meta_path = os.path.join(index_dir, "meta.json")
#     if os.path.exists(meta_path):
#         with open(meta_path, "r", encoding="utf-8") as f:
#             meta = json.load(f)
#         model_name = meta.get("model", fallback_model)
#     else:
#         model_name = fallback_model
#
#     emb = get_embedder(model_name)
#     vs = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
#     return vs, model_name
#
# def query(index_dir: str, question: str, k: int, model_name: str) -> List[Tuple[str, float, dict]]:
#     vs, used_model = load_index(index_dir, model_name)
#     results = vs.similarity_search_with_score(question, k=k)
#     # NOTE: FAISS returns distance; with normalized embeddings, smaller is better.
#     print(f"[Ask] Using embedding model: {used_model}")
#     return [(doc.page_content, score, doc.metadata) for (doc, score) in results]
#
# def main():
#     parser = argparse.ArgumentParser(description="Simple sentence-level RAG (LangChain + FAISS).")
#     sub = parser.add_subparsers(dest="cmd", required=True)
#
#     p_index = sub.add_parser("index", help="Lập chỉ mục từ 1 file .txt")
#     p_index.add_argument("--input", required=True, help="Đường dẫn file .txt")
#     p_index.add_argument("--index-dir", default="index.faiss", help="Thư mục lưu FAISS")
#     p_index.add_argument("--model", default=DEFAULT_MODEL, help="Tên model sentence-transformers")
#
#     p_ask = sub.add_parser("ask", help="Đặt câu hỏi và retrieve các câu gần nghĩa")
#     p_ask.add_argument("--index-dir", default="index.faiss", help="Thư mục FAISS đã lưu")
#     p_ask.add_argument("--q", required=True, help="Câu hỏi / truy vấn")
#     p_ask.add_argument("--k", type=int, default=5, help="Số câu cần lấy")
#     # This is only a fallback if meta.json is missing or edited
#     p_ask.add_argument("--model", default=DEFAULT_MODEL, help="Tên model sentence-transformers (fallback)")
#
#     args = parser.parse_args()
#
#     if args.cmd == "index":
#         index_corpus(args.input, args.index_dir, args.model)
#
#     elif args.cmd == "ask":
#         hits = query(args.index_dir, args.q, args.k, args.model)
#         print(f"\n[Query] {args.q}\n")
#         for rank, (sent, score, meta) in enumerate(hits, 1):
#             # print(f"#{rank}  score={score:.4f}  source={meta.get('source')}  id={meta.get('sentence_id')}")
#             print(f"{sent}\n")
#
# if __name__ == "__main__":
#     main()
# rag_folder.py

import argparse, os, json
from pathlib import Path
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Config ----
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SIMILARITY = 0.40   # cosine threshold to print (adjust 0.35–0.45)

# ---- Embeddings ----
def embedder(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 128}
    )

# ---- Text split (LangChain) ----
def make_splitter() -> RecursiveCharacterTextSplitter:
    """
    Sentence-ish splitting with graceful fallback.
    - Prefer splitting on sentence ends and newlines.
    - Then on bullets/dashes/semicolons.
    - Finally long words if needed.
    """
    return RecursiveCharacterTextSplitter(
        separators=[
            # sentence-ish first
            ". ", "? ", "! ", "… ",
            # paragraph / newlines
            "\n\n", "\n",
            # bullets and commas/semicolons
            " • ", " - ", " – ", "; ",
            # final fallback
            " "
        ],
        chunk_size=300,       # ~1–2 sentences each (tweak as you like)
        chunk_overlap=30,     # small overlap to avoid cutting ideas
        length_function=len,
        keep_separator=False
    )

def split_into_chunks(text: str) -> List[str]:
    splitter = make_splitter()
    chunks = splitter.split_text(text or "")
    # filter tiny crumbs
    return [c.strip() for c in chunks if len(c.strip()) >= 20]

# ---- IO ----
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def build_docs(chunks: List[str], src: Path) -> List[Document]:
    return [
        Document(page_content=c, metadata={"source": src.name, "source_path": str(src), "chunk_id": i})
        for i, c in enumerate(chunks)
    ]

# ---- Meta ----
def save_meta(index_dir: Path, model_name: str, files: List[str], num_docs: int):
    meta = {"model": model_name, "files": sorted(files), "num_documents": num_docs}
    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def load_meta(index_dir: Path) -> dict:
    p = index_dir / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

# ---- Indexing ----
def index_folder(folder: Path, index_dir: Path, model_name: str, pattern: str = "*.txt", recursive: bool = True):
    files = list(folder.rglob(pattern) if recursive else folder.glob(pattern))
    files = [f for f in files if f.is_file()]
    if not files:
        raise ValueError("Không tìm thấy file .txt nào trong folder.")

    emb = embedder(model_name)
    vs = None
    total_docs, indexed_files = 0, []

    for f in files:
        text = read_text(f)
        chunks = split_into_chunks(text)
        if not chunks:
            continue
        docs = build_docs(chunks, f)
        if vs is None:
            vs = FAISS.from_documents(docs, emb)
        else:
            vs.add_documents(docs)
        total_docs += len(docs)
        indexed_files.append(str(f))

    if vs is None:
        raise ValueError("Tất cả file đều rỗng.")

    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    save_meta(index_dir, model_name, indexed_files, total_docs)
    print(f"[Index] {total_docs} đoạn từ {len(indexed_files)} file. Lưu tại: {index_dir}")

# ---- Load ----
def load_index(index_dir: Path, fallback_model: str) -> Tuple[FAISS, str]:
    meta = load_meta(index_dir)
    model_name = meta.get("model", fallback_model)
    emb = embedder(model_name)
    vs = FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)
    return vs, model_name

# ---- Ask top-k ----
def ask_topk(index_dir: Path, query: str, model_name: str, k: int = 5) -> Optional[List[Tuple[str, float, dict]]]:
    vs, used_model = load_index(index_dir, model_name)
    results = vs.similarity_search_with_score(query, k=k)
    if not results:
        return None

    # FAISS distance (on normalized vectors) -> cosine similarity: cos = 1 - d/2
    hits = []
    for doc, dist in results:
        cos = 1.0 - float(dist) / 2.0
        if cos >= MIN_SIMILARITY:
            hits.append((doc.page_content, float(dist), doc.metadata))

    if not hits:
        return None

    # Print top-k chunks (sentence-ish)
    for sent, _, _ in hits:
        print(sent)
    return hits

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="RAG: index folder .txt (LC TextSplitter) & print top-5 chunks.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index-folder", help="Lập chỉ mục cả thư mục .txt")
    p_idx.add_argument("--folder", required=True, help="Thư mục chứa .txt")
    p_idx.add_argument("--index-dir", default="index.faiss", help="Nơi lưu FAISS")
    p_idx.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model")

    p_ask = sub.add_parser("ask", help="Hỏi và in 5 đoạn gần nghĩa nhất")
    p_ask.add_argument("--index-dir", default="index.faiss", help="Thư mục FAISS đã lưu")
    p_ask.add_argument("--q", required=True, help="Câu hỏi")
    p_ask.add_argument("--k", type=int, default=5, help="Số đoạn cần lấy (mặc định 5)")
    p_ask.add_argument("--model", default=DEFAULT_MODEL, help="fallback model nếu thiếu meta")

    args = ap.parse_args()
    if args.cmd == "index-folder":
        index_folder(Path(args.folder), Path(args.index_dir), args.model)
    elif args.cmd == "ask":
        ask_topk(Path(args.index_dir), args.q, args.model, k=args.k)

if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from pymilvus import MilvusClient, model

INDEX_DATA_PATH = "./.db"
INDEX_TRACKING_FILE = "index_tracking.json"
COLLECTION_NAME = "markdown_vectors"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("markdown-rag")


def _load_tracking():
    try:
        with open(os.path.join(INDEX_DATA_PATH, INDEX_TRACKING_FILE)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_tracking(data):
    with open(os.path.join(INDEX_DATA_PATH, INDEX_TRACKING_FILE), "w") as f:
        json.dump(data, f, indent=2)


def _file_info(path):
    with open(path, "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()
    return h, os.path.getmtime(path)


def _update_tracking(files, clear=False):
    data = _load_tracking()
    if clear:
        _save_tracking({})
        return
    for p in files:
        try:
            data[p] = _file_info(p)
        except (FileNotFoundError, PermissionError):
            data.pop(p, None)
    _save_tracking(data)


def _list_md(base, recursive):
    result = []
    if recursive:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            result += [os.path.join(root, f) for f in files if f.endswith(".md")]
    else:
        result = [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".md")]
    return result


def _changed_files(directory, recursive):
    tracking = _load_tracking()
    changed = []
    for path in _list_md(directory, recursive):
        if path not in tracking:
            changed.append(path)
            continue
        try:
            h, t = _file_info(path)
        except (FileNotFoundError, PermissionError):
            continue
        sh, st = tracking[path]
        if h != sh or t != st:
            changed.append(path)
    return changed


def _ensure_collection(client):
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(COLLECTION_NAME, dimension=768, auto_id=True)


def run(directory: str, recursive: bool, force_reindex: bool):
    if not os.path.exists(INDEX_DATA_PATH):
        os.makedirs(INDEX_DATA_PATH)

    log.info("Connecting to Milvus Lite...")
    client = MilvusClient(os.path.join(INDEX_DATA_PATH, "milvus_markdown.db"))

    log.info("Loading embedding model (all-MiniLM-L6-v2)...")
    t0 = time.time()
    embedding_fn = model.DefaultEmbeddingFunction()
    log.info(f"Embedding model ready in {time.time() - t0:.1f}s")

    if force_reindex:
        log.info("Force reindex — dropping existing collection...")
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
        _ensure_collection(client)

        files = _list_md(directory, recursive)
        log.info(f"Found {len(files)} markdown file(s)")
        for f in files:
            log.info(f"  {f}")
        docs = SimpleDirectoryReader(input_files=files, required_exts=[".md"]).load_data()
        processed = [d.metadata["file_path"] for d in docs]
    else:
        log.info("Scanning for changed files...")
        changed = _changed_files(directory, recursive)

        if not changed:
            log.info("Already up to date — nothing to index.")
            return

        tracking = _load_tracking()
        new_files = [f for f in changed if f not in tracking]
        modified_files = [f for f in changed if f in tracking]

        if new_files:
            log.info(f"{len(new_files)} new file(s):")
            for f in new_files:
                log.info(f"  {f}")
        if modified_files:
            log.info(f"{len(modified_files)} modified file(s):")
            for f in modified_files:
                log.info(f"  {f}")

        _ensure_collection(client)

        if modified_files:
            log.info("Removing stale chunks for modified files...")
            for path in modified_files:
                try:
                    res = client.delete(COLLECTION_NAME, filter=f"path == '{path}'")
                    log.info(f"  Deleted old chunks for {os.path.basename(path)}: {res}")
                except Exception as e:
                    log.warning(f"  Could not delete chunks for {path}: {e}")

        docs = SimpleDirectoryReader(input_files=changed, required_exts=[".md"]).load_data()
        processed = changed

    log.info(f"Loaded {len(docs)} document(s) — parsing markdown structure...")
    nodes = MarkdownNodeParser().get_nodes_from_documents(docs)
    log.info(f"  {len(nodes)} node(s) after markdown parsing")

    chunked = TokenTextSplitter(chunk_size=512, chunk_overlap=100).get_nodes_from_documents(nodes)
    chunked = [n for n in chunked if n.text.strip()]
    log.info(f"  {len(chunked)} chunk(s) after splitting (512 tokens, 100 overlap)")

    file_chunks: dict[str, list] = {}
    for n in chunked:
        file_chunks.setdefault(n.metadata["file_path"], []).append(n)

    total_files = len(file_chunks)
    total_chunks = len(chunked)
    log.info(f"Embedding and inserting {total_chunks} chunk(s) from {total_files} file(s)...")
    t0 = time.time()

    ordered_nodes = []
    for i, (path, nodes) in enumerate(file_chunks.items(), 1):
        log.info(f"  [{i}/{total_files}] {os.path.basename(path)} ({len(nodes)} chunk(s))")
        vecs = embedding_fn.encode_documents([n.text for n in nodes])
        data = [
            {
                "vector": v,
                "text": n.text,
                "filename": n.metadata["file_name"],
                "path": n.metadata["file_path"],
            }
            for v, n in zip(vecs, nodes)
        ]
        client.insert(COLLECTION_NAME, data)
        _update_tracking([path])
        ordered_nodes.extend(nodes)

    log.info(f"  Done in {time.time() - t0:.1f}s")

    log.info("")
    log.info(f"Done: {len(processed)} file(s), {len(ordered_nodes)} chunk(s) total")
    log.info("Chunks per file:")
    for name, count in sorted(Counter(n.metadata["file_name"] for n in ordered_nodes).items()):
        log.info(f"  {name}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Index markdown files for semantic search (MCP Markdown RAG)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="",
        help="Directory to index (default: current working directory)",
    )
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--force", action="store_true", help="Force full reindex")
    args = parser.parse_args()

    target = os.path.abspath(os.path.join(os.getcwd(), args.directory))
    if not os.path.exists(target):
        log.error(f"Directory not found: {target}")
        sys.exit(1)

    log.info(f"Directory : {target}")
    log.info(f"Recursive : {args.recursive}")
    log.info(f"Force     : {args.force}")

    run(target, recursive=args.recursive, force_reindex=args.force)


if __name__ == "__main__":
    main()

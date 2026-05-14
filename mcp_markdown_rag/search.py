import argparse
import os
import sys

from pymilvus import MilvusClient, model

INDEX_DATA_PATH = "./.db"
COLLECTION_NAME = "markdown_vectors"


def main():
    parser = argparse.ArgumentParser(
        description="Search indexed markdown documents (MCP Markdown RAG)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    db_path = os.path.join(INDEX_DATA_PATH, "milvus_markdown.db")
    if not os.path.exists(db_path):
        print("No index found. Run `markdown-rag-index` first.", file=sys.stderr)
        sys.exit(1)

    client = MilvusClient(db_path)

    if not client.has_collection(COLLECTION_NAME):
        print("Index is empty. Run `markdown-rag-index` first.", file=sys.stderr)
        sys.exit(1)

    client.load_collection(COLLECTION_NAME)

    embedding_fn = model.DefaultEmbeddingFunction()
    query_vectors = embedding_fn.encode_queries([args.query])

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vectors,
        limit=args.k,
        output_fields=["text", "filename", "path"],
    )

    if not results or not results[0]:
        print("No results found.")
        return

    for i, hit in enumerate(results[0], 1):
        e = hit["entity"]
        score = hit["distance"]
        print(f"\n{'─' * 60}")
        print(f"[{i}/{args.k}] {e['filename']}  (score: {score:.4f})")
        print(f"     {e['path']}")
        print(f"{'─' * 60}")
        print(e["text"])

    print()


if __name__ == "__main__":
    main()

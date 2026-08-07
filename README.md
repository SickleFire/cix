# CIX

cix (Code Indexer) is a fast, lightweight command-line code search engine built in Rust. Powered by Tantivy, cix pre-indexes your source code using BM25 relevance scoring to deliver sub-millisecond full-text search results across local codebases.

Features
- BM25 Relevance Ranking: Search results are ranked by term frequency and relevance rather than dumped as unranked line matches.

-  Incremental Indexing: Tracks file modification timestamps (mtime) to index only files that have changed since the last run.

- Duplicate-Safe Upserts: Uses exact-match path terms (delete_term) to update modified files without duplicating documents in the index.

- Persistent Disk Storage: Stores memory-mapped inverted indexes (MmapDirectory) on disk for instant subsequent lookups.

## Usage
`cix <search_query> <target_directory>`

<img width="622" height="176" alt="image" src="https://github.com/user-attachments/assets/753df8bf-291c-437f-b6d8-ae919c25560e" />


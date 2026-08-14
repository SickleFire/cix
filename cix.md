# CIX (Code Indexer)

`cix` is a fast, lightweight command-line code search engine built in Rust. Powered by Tantivy, `cix` pre-indexes your source code using BM25 relevance scoring to deliver sub-millisecond full-text search results across local codebases.

## Features

- **BM25 Relevance Ranking**: Search results are ranked by term frequency and relevance rather than dumped as unranked line matches.
- **Incremental Indexing**: Tracks file modification timestamps (`mtime`) to index only files that have changed since the last run.
- **Duplicate-Safe Upserts**: Uses exact-match path terms (`delete_term`) to update modified files without duplicating documents in the index.
- **Persistent Disk Storage**: Stores memory-mapped inverted indexes (`MmapDirectory`) on disk for instant subsequent lookups.

## Usage

```bash
cix <search_query> <target_directory>
```

## Dependencies

- [Tantivy](https://github.com/quickwit-oss/tantivy): Full-text search engine library
- [Clap](https://github.com/clap-rs/clap): Command line argument parser
- [Ignore](https://github.com/BurntSushi/ripgrep/tree/master/crates/ignore): Fast file traversal respecting `.gitignore`
- [Tokio](https://tokio.rs/) & [Reqwest](https://github.com/seanmonstar/reqwest): Async runtime and HTTP client

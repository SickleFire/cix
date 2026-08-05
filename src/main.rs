use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, ReloadPolicy, TantivyDocument, Term, doc};
use walkdir::WalkDir;

fn main() -> tantivy::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <target_directory> <search_query>", args[0]);
        std::process::exit(1);
    }
    let query_target_directory = &args[1];
    let query_arg = &args[2];

    let mut schema_builder = Schema::builder();
    let file_path = schema_builder.add_text_field("path", STRING | STORED);
    let content = schema_builder.add_text_field("content", TEXT | STORED);
    let schema = schema_builder.build();

    let index_path = "./my_index";
    fs::create_dir_all(index_path).unwrap_or_default();
    let dir = tantivy::directory::MmapDirectory::open(index_path)?;
    let index = Index::open_or_create(dir, schema.clone())?;

    let mut index_writer = index.writer(100_000_000)?;
    let target_directory = query_target_directory;

    let mut hasher = DefaultHasher::new();
    target_directory.hash(&mut hasher);
    let dir_hash = hasher.finish();

    let state_file = format!("{}/.last_run_{}", index_path, dir_hash);

    let last_run: u64 = fs::read_to_string(&state_file)
        .unwrap_or_else(|_| "0".to_string())
        .trim()
        .parse()
        .unwrap_or(0);

    let current_run = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    println!(
        "Indexing files in {} modified after timestamp {}...",
        target_directory, last_run
    );
    let mut updated_count = 0;

    for entry in WalkDir::new(target_directory)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            let path_str = entry.path().to_string_lossy().to_string();

            // Only index Rust files for now
            if is_indexable_file(&path_str) {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified_time) = metadata.modified() {
                        let modified_secs =
                            modified_time.duration_since(UNIX_EPOCH).unwrap().as_secs();

                        if modified_secs >= last_run {
                            if let Ok(file_content) = fs::read_to_string(entry.path()) {
                                let path_term = Term::from_field_text(file_path, &path_str);

                                index_writer.delete_term(path_term);

                                index_writer.add_document(doc!(
                                    file_path => path_str,
                                    content => file_content
                                ))?;

                                updated_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    index_writer.commit()?;
    fs::write(state_file, current_run.to_string())?;

    println!("Indexing complete! {} files updated.", updated_count);

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    let query_parser = QueryParser::for_index(&index, vec![content]);
    let query = query_parser.parse_query(query_arg)?;

    let top_docs = searcher.search(&query, &TopDocs::with_limit(5).and_offset(0))?;

    println!("\nFound {} results for '{}':", top_docs.len(), query_arg);
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let path_val = retrieved_doc.get_first(file_path).unwrap().as_str().unwrap();
        let content_val = retrieved_doc.get_first(content).unwrap().as_str().unwrap();

        println!("\nScore: {:.2} | File: {}", score, path_val);

        // Find and print specific matching lines with line numbers
        let query_lower = query_arg.to_lowercase();
        for (line_idx, line) in content_val.lines().enumerate() {
            if line.to_lowercase().contains(&query_lower) {
                println!("  Line {:>4}: {}", line_idx + 1, line.trim());
            }
        }
    }

    Ok(())
}

fn is_indexable_file(path_str: &str) -> bool {
    let allowed_extensions = [
        "rs", "c", "cpp", "h", "hpp", "py", "js", "ts", 
        "toml", "json", "yaml", "yml", "md", "txt", "sh"
    ];

    std::path::Path::new(path_str)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| allowed_extensions.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

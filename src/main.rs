use clap::Parser;
use colored::*;
use ignore::WalkBuilder;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, ReloadPolicy, TantivyDocument, Term, doc};

#[derive(Parser, Debug)]
#[command(
    name = "cix",
    author,
    version,
    about = "Instant BM25 indexed code search engine"
)]
struct Cli {
    /// Search query term
    #[arg(required_unless_present = "clean")]
    search_query: Option<String>,

    /// Target directory to index and search (defaults to current directory)
    #[arg(default_value = ".")]
    target_directory: String,

    /// Number of context lines to display
    #[arg(short = 'C', long, default_value_t = 1)]
    context: usize,

    /// Maximum number of top matching files to return
    #[arg(short = 'l', long, default_value_t = 10)]
    limit: usize,

    /// Force a full re-index
    #[arg(short = 'r', long, default_value_t = false)]
    reindex: bool,

    /// Remove all cached indexes
    #[arg(long, default_value_t = false)]
    clean: bool,
}

fn main() -> tantivy::Result<()> {
    let cli = Cli::parse();

    // 1. Setup Base Cache Directory
    let cache_dir = dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let app_cache_dir = cache_dir.join("cix_indexes");

    // Handle --clean option early
    if cli.clean {
        if app_cache_dir.exists() {
            match fs::remove_dir_all(&app_cache_dir) {
                Ok(_) => println!(
                    "{}",
                    "Successfully cleared all cix cache indexes!".green().bold()
                ),
                Err(e) => eprintln!("Failed to remove cache directory: {}", e),
            }
        } else {
            println!("{}", "No cache directory found to clean.".yellow());
        }
        return Ok(());
    }

    // Unwrap arguments (clap guarantees present when --clean is false)
    let query_target_directory = &cli.target_directory;
    let query_arg = cli.search_query.as_ref().unwrap();
    let context_size = cli.context;
    let result_limit = cli.limit;

    let target_path = fs::canonicalize(query_target_directory)
        .unwrap_or_else(|_| std::path::PathBuf::from(query_target_directory));

    // Hash directory path for index key
    let mut hasher = DefaultHasher::new();
    target_path.hash(&mut hasher);
    let dir_hash = hasher.finish();

    let index_path = app_cache_dir.join(format!("{}", dir_hash));

    // Handle --reindex flag: remove single index directory
    if cli.reindex && index_path.exists() {
        println!(
            "{}",
            "Forcing full re-index: purging directory cache..."
                .yellow()
                .bold()
        );
        if let Err(e) = fs::remove_dir_all(&index_path) {
            eprintln!("Failed to purge directory cache: {}", e);
        }
    }

    fs::create_dir_all(&index_path).unwrap_or_default();

    let state_file = index_path.join(".last_run");

    let last_run: u64 = if cli.reindex {
        0
    } else {
        fs::read_to_string(&state_file)
            .unwrap_or_else(|_| "0".to_string())
            .trim()
            .parse()
            .unwrap_or(0)
    };

    let current_run = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    println!(
        "Indexing files in {} modified after timestamp {}...",
        target_path.display(),
        last_run
    );
    let mut updated_count = 0;

    // 2. Setup Tantivy Schema
    let mut schema_builder = Schema::builder();
    let file_path = schema_builder.add_text_field("path", STRING | STORED);
    let content = schema_builder.add_text_field("content", TEXT | STORED);
    let schema = schema_builder.build();

    let dir = tantivy::directory::MmapDirectory::open(&index_path)?;
    let index = Index::open_or_create(dir, schema.clone())?;
    let mut index_writer = index.writer(100_000_000)?;

    // 3. Crawl Files
    for result in WalkBuilder::new(&target_path).hidden(true).build() {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };

        if entry.file_type().map_or(false, |ft| ft.is_file()) {
            let path_str = entry.path().to_string_lossy().to_string();

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

    // 4. Search and Display
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    let query_parser = QueryParser::for_index(&index, vec![content]);
    let query = match query_parser.parse_query(query_arg) {
        Ok(q) => q,
        Err(_) => {
            eprintln!("Invalid query format.");
            return Ok(());
        }
    };

    let top_docs = searcher.search(&query, &TopDocs::with_limit(result_limit).and_offset(0))?;

    println!(
        "\nFound {} results for '{}':",
        top_docs.len().to_string().bold().cyan(),
        query_arg.bold().yellow()
    );

    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let path_val = retrieved_doc
            .get_first(file_path)
            .unwrap()
            .as_str()
            .unwrap();
        let content_val = retrieved_doc.get_first(content).unwrap().as_str().unwrap();

        println!(
            "\n{} {} | {}: {}",
            "Score:".dimmed(),
            format!("{:.2}", score).cyan().bold(),
            "File".dimmed(),
            path_val.bold().green()
        );

        let lines: Vec<&str> = content_val.lines().collect();
        let total_lines = lines.len();
        let query_lower = query_arg.to_lowercase();

        let matching_indices: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter_map(|(idx, line): (usize, &&str)| {
                if line.to_lowercase().contains(&query_lower) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        if matching_indices.is_empty() {
            continue;
        }

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for &idx in &matching_indices {
            let idx: usize = idx; // Explicit type annotation
            let start = idx.saturating_sub(context_size);
            let end = (idx + context_size).min(total_lines.saturating_sub(1));

            if let Some(last) = ranges.last_mut() {
                if start <= last.1 + 1 {
                    last.1 = last.1.max(end);
                } else {
                    ranges.push((start, end));
                }
            } else {
                ranges.push((start, end));
            }
        }

        for (range_idx, &(start, end)) in ranges.iter().enumerate() {
            if range_idx > 0 {
                println!("  {}", "--".dimmed());
            }

            for i in start..=end {
                let line_str = lines[i];
                let is_match = matching_indices.contains(&i);

                if is_match {
                    let line_num = format!("Line {:>4}:", i + 1).blue().bold();
                    let highlighted = highlight_match(line_str, query_arg);
                    println!("  {} {}", line_num, highlighted);
                } else {
                    let line_num = format!("Line {:>4}:", i + 1).dimmed();
                    println!("  {} {}", line_num, line_str.dimmed());
                }
            }
        }
    }

    Ok(())
}

fn is_indexable_file(path_str: &str) -> bool {
    let allowed_extensions = [
        "rs", "c", "cpp", "h", "hpp", "py", "js", "ts", "toml", "json", "yaml", "yml", "md", "txt",
        "sh", "cs",
    ];

    std::path::Path::new(path_str)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| allowed_extensions.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn highlight_match(line: &str, query: &str) -> String {
    if query.is_empty() {
        return line.to_string();
    }

    let mut highlighted = String::new();
    let line_lower = line.to_lowercase();
    let query_lower = query.to_lowercase();
    let query_len = query.len();

    let mut last_end = 0;
    for (start, _) in line_lower.match_indices(&query_lower) {
        highlighted.push_str(&line[last_end..start]);
        let matched_text = &line[start..start + query_len];
        highlighted.push_str(&matched_text.bold().red().to_string());
        last_end = start + query_len;
    }
    highlighted.push_str(&line[last_end..]);
    highlighted
}

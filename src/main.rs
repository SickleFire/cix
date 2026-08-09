//! # cix - Indexed Code Search & RAG Tool
//!
//! `cix` is a CLI tool providing fast indexed code search, codebase RAG question-answering (`cix ask`),
//! and AI-driven automated code modification (`cix edit`).

use clap::{Parser, Subcommand};
use colored::*;
use futures_util::StreamExt;
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
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
    about = "Indexed Code Search & RAG Tool"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query term (when not using a subcommand)
    query: Option<String>,

    /// Target directory to index and search
    #[arg(default_value = ".")]
    target_directory: String,

    /// Number of context lines to display for code matches
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

#[derive(Subcommand, Debug)]
enum Commands {
    /// Ask an LLM a question about your codebase using retrieved context
    Ask {
        /// The question to ask about your codebase
        question: String,

        /// Model identifier
        #[arg(
            short = 'm',
            long,
            env = "CIX_MODEL",
            default_value = "gemini-3.6-flash"
        )]
        model: String,

        /// Provider override: 'ollama', 'gemini', or 'auto'
        #[arg(short = 'p', long, default_value = "gemini")]
        provider: String,

        /// Target directory
        #[arg(default_value = ".")]
        target_directory: String,
    },

    /// Request the AI to modify files in your codebase
    Edit {
        /// Instruction for the code modification
        instruction: String,

        /// Model identifier
        #[arg(
            short = 'm',
            long,
            env = "CIX_MODEL",
            default_value = "gemini-3.6-flash"
        )]
        model: String,

        /// Provider override: 'ollama', 'gemini', or 'auto'
        #[arg(short = 'p', long, default_value = "gemini")]
        provider: String,

        /// Target directory
        #[arg(default_value = ".")]
        target_directory: String,
    },
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    /// JSON schema for constrained decoding. `None` for free-form answers
    /// (e.g. `cix ask`), `Some(schema)` when the response must match a
    /// structural contract (e.g. `cix edit`'s edit-block list). Skipped
    /// entirely when absent so Ollama sees no `format` field at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    done: bool,
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Deserialize, Clone)]
struct GeminiCandidate {
    content: GeminiResponseContent,
}

#[derive(Deserialize, Clone)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize, Clone)]
struct GeminiResponsePart {
    text: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // 1. Handle --clean flag
    let cache_dir = dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let app_cache_dir = cache_dir.join("cix_indexes");

    if cli.clean {
        if app_cache_dir.exists() {
            fs::remove_dir_all(&app_cache_dir)?;
            println!(
                "{}",
                "Successfully cleared all cix cache indexes!".green().bold()
            );
        } else {
            println!("{}", "No cache directory found to clean.".yellow());
        }
        return Ok(());
    }

    // 2. Handle 'cix ask' Subcommand
    if let Some(Commands::Ask {
        question,
        model,
        provider,
        target_directory,
    }) = &cli.command
    {
        run_ask_pipeline(question, model, provider, target_directory, &app_cache_dir).await?;
        return Ok(());
    }

    // 3. Handle 'cix edit' Subcommand
    if let Some(Commands::Edit {
        instruction,
        model,
        provider,
        target_directory,
    }) = &cli.command
    {
        run_edit_pipeline(
            instruction,
            model,
            provider,
            target_directory,
            &app_cache_dir,
        )
        .await?;
        return Ok(());
    }

    // 4. Fallback to standard search if no subcommand
    if let Some(query_str) = &cli.query {
        run_search_pipeline(
            query_str,
            &cli.target_directory,
            cli.context,
            cli.limit,
            cli.reindex,
            &app_cache_dir,
        )?;
    } else {
        println!(
            "{}",
            "Please provide a search query or run 'cix ask \"<question>\"'".yellow()
        );
    }

    Ok(())
}

/// Extracts search keywords from a user prompt or question by removing common stop words
/// and standard file extensions to optimize indexing queries.
fn extract_keywords(question: &str) -> String {
    let stop_words = [
        "what", "does", "how", "is", "are", "the", "a", "an", "do", "can", "you", "tell", "me",
        "about", "file", "script", "code", "in", "of", "for", "to", "with", "work", "where",
        "which", "show", "explain", "meaning",
    ];

    let extensions = [
        ".cs", ".rs", ".py", ".js", ".ts", ".cpp", ".h", ".json", ".toml", ".yaml", ".md",
        ".shader", ".hlsl",
    ];

    let mut cleaned_question = question.to_string();
    for ext in extensions {
        cleaned_question = cleaned_question.replace(ext, "");
    }

    let keywords: Vec<&str> = cleaned_question
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|word| {
            let w = word.to_lowercase();
            !w.is_empty() && !stop_words.contains(&w.as_str())
        })
        .collect();

    if keywords.is_empty() {
        question.to_string()
    } else {
        keywords.join(" ")
    }
}

/// Executes the RAG question-answering pipeline: retrieves relevant file contexts
/// from the index and queries Gemini API (or local Ollama as fallback).
async fn run_ask_pipeline(
    question: &str,
    model: &str,
    provider: &str,
    target_dir: &str,
    app_cache_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "Retrieving codebase context...".cyan());

    let (index, file_path_field, content_field) = ensure_index(target_dir, false, app_cache_dir)?;
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    // Search across both file path and content fields
    let query_parser = QueryParser::for_index(&index, vec![file_path_field, content_field]);
    let query = parse_query_with_fuzzy(&query_parser, question);

    let mut top_docs = searcher.search(&query, &TopDocs::with_limit(4).and_offset(0))?;

    // Fallback: If generic question returns no hits, pull top indexed files
    if top_docs.is_empty() {
        if let Ok(fallback_query) = query_parser.parse_query("*") {
            top_docs = searcher.search(&fallback_query, &TopDocs::with_limit(4).and_offset(0))?;
        }
    }

    if top_docs.is_empty() {
        println!("{}", "No relevant code files found for context.".yellow());
        return Ok(());
    }

    let mut context_payload = String::new();
    let mut seen_paths = HashSet::new();
    println!("{}", "Found relevant context in:".dimmed());

    for (_, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let path = retrieved_doc
            .get_first(file_path_field)
            .unwrap()
            .as_str()
            .unwrap();

        // Skip duplicate hits for the same file so it isn't fed to the model
        // multiple times (wastes context budget and confuses smaller models).
        if !seen_paths.insert(path.to_string()) {
            continue;
        }

        let content = retrieved_doc
            .get_first(content_field)
            .unwrap()
            .as_str()
            .unwrap();

        println!("  • {}", path.bold().green());
        context_payload.push_str(&format!("\n--- FILE: {} ---\n{}\n", path, content));
    }

    let prompt = format!(
        "You are an expert software engineer inspecting a codebase.\n\
        Use the following retrieved code files to answer the user's question accurately.\n\
        If the answer is not in the code provided, state what is missing.\n\n\
        CODEBASE CONTEXT:\n{}\n\n\
        USER QUESTION:\n{}\n\n\
        ANSWER:",
        context_payload, question
    );

    let client = reqwest::Client::new();

    // Determine provider based on explicit flag or model name substring
    let use_gemini = match provider.to_lowercase().as_str() {
        "gemini" => true,
        "ollama" | "local" => false,
        _ => model.to_lowercase().contains("gemini"),
    };

    if use_gemini {
        let api_key = std::env::var("GEMINI_API_KEY")
            .map_err(|_| "GEMINI_API_KEY environment variable is missing.")?;

        let model_name = model.trim_start_matches("models/");

        println!(
            "\n{}",
            format!(" Asking Gemini API ({}) ...", model_name)
                .bold()
                .cyan()
        );
        println!(
            "{}",
            "--------------------------------------------------".dimmed()
        );

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model_name, api_key
        );

        let body = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart { text: prompt }],
            }],
        };

        let res = client.post(&url).json(&body).send().await?;

        if res.status().is_success() {
            let gemini_res: GeminiResponse = res.json().await?;
            if let Some(candidates) = gemini_res.candidates {
                if let Some(first_candidate) = candidates.first() {
                    for part in &first_candidate.content.parts {
                        println!("{}", part.text);
                    }
                }
            }
        } else {
            let err_text = res.text().await?;
            eprintln!(
                "{}",
                format!("\n Gemini API Error: {}", err_text).red().bold()
            );
        }
    } else {
        // Local streaming Ollama provider
        println!(
            "\n{}",
            format!(" Asking local Ollama ({}) ...", model)
                .bold()
                .magenta()
        );
        println!(
            "{}",
            "--------------------------------------------------".dimmed()
        );

        let res = client
            .post("http://localhost:11434/api/generate")
            .json(&OllamaRequest {
                model: model.to_string(),
                prompt,
                stream: true,
                // `ask` is free-form Q&A — no schema constraint. Forcing it
                // into the edit-block JSON shape would make every answer
                // garbage or empty.
                format: None,
            })
            .send()
            .await;

        match res {
            Ok(response) => {
                let mut stream = response.bytes_stream();
                let mut stdout = std::io::stdout();

                while let Some(chunk) = stream.next().await {
                    if let Ok(bytes) = chunk {
                        let line = String::from_utf8_lossy(&bytes);
                        for sub_line in line.lines() {
                            if let Ok(parsed) = serde_json::from_str::<OllamaResponse>(sub_line) {
                                print!("{}", parsed.response);
                                stdout.flush()?;
                                if parsed.done {
                                    println!();
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            Err(_) => {
                eprintln!(
                    "{}",
                    "\n Failed to connect to local Ollama server at http://localhost:11434."
                        .red()
                        .bold()
                );
            }
        }
    }

    Ok(())
}

/// Executes the code search pipeline: queries the Tantivy index and displays matching
/// code snippets with line numbers and context preview.
fn run_search_pipeline(
    query_arg: &str,
    target_directory: &str,
    context_size: usize,
    result_limit: usize,
    reindex: bool,
    app_cache_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (index, file_path, content) = ensure_index(target_directory, reindex, app_cache_dir)?;

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    let query_parser = QueryParser::for_index(&index, vec![file_path, content]);
    let query = parse_query_with_fuzzy(&query_parser, query_arg);

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
        let keywords: Vec<String> = extract_keywords(query_arg)
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        let mut matching_indices: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter_map(|(idx, line): (usize, &&str)| {
                let line_lower = line.to_lowercase();
                if keywords.iter().any(|kw| line_matches_keyword(&line_lower, kw)) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        if matching_indices.is_empty() && !lines.is_empty() {
            matching_indices.push(0);
        }

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for &idx in &matching_indices {
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

struct EditBlock {
    file_path: String,
    search: String,
    replace: String,
}

/// Executes the code editing pipeline: retrieves relevant code context, requests
/// structured edit blocks from Gemini (marker-format text) or Ollama
/// (schema-constrained JSON), and applies changes on confirmation.
async fn run_edit_pipeline(
    instruction: &str,
    model: &str,
    provider: &str,
    target_dir: &str,
    app_cache_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "Retrieving relevant context for edit...".cyan());

    let (index, file_path_field, content_field) = ensure_index(target_dir, false, app_cache_dir)?;
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    let query_parser = QueryParser::for_index(&index, vec![file_path_field, content_field]);
    let query = parse_query_with_fuzzy(&query_parser, instruction);

    let mut top_docs = searcher.search(&query, &TopDocs::with_limit(3).and_offset(0))?;

    if top_docs.is_empty() {
        if let Ok(fallback_query) = query_parser.parse_query("*") {
            top_docs = searcher.search(&fallback_query, &TopDocs::with_limit(3).and_offset(0))?;
        }
    }

    if top_docs.is_empty() {
        println!("{}", "No relevant code files found to modify.".yellow());
        return Ok(());
    }

    let mut context_payload = String::new();
    let mut seen_paths = HashSet::new();
    let mut example_snippet: Option<(String, String)> = None; // (path, first_line) for Gemini's few-shot example
    println!("{}", "Target files for context:".dimmed());

    for (_, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let path = retrieved_doc
            .get_first(file_path_field)
            .unwrap()
            .as_str()
            .unwrap();

        // Skip duplicate hits for the same file.
        if !seen_paths.insert(path.to_string()) {
            continue;
        }

        let content = retrieved_doc
            .get_first(content_field)
            .unwrap()
            .as_str()
            .unwrap();

        if example_snippet.is_none() {
            if let Some(first_line) = content.lines().find(|l| !l.trim().is_empty()) {
                example_snippet = Some((path.to_string(), first_line.to_string()));
            }
        }

        println!("  • {}", path.bold().green());
        context_payload.push_str(&format!("\n--- FILE: {} ---\n{}\n", path, content));
    }

    let client = reqwest::Client::new();
    let use_gemini = match provider.to_lowercase().as_str() {
        "gemini" => true,
        "ollama" | "local" => false,
        _ => model.to_lowercase().contains("gemini"),
    };

    // Gemini has no constrained-decoding hookup here, so it still needs the
    // marker-format prompt (with a grounded few-shot example — small/medium
    // models imitate an abstract placeholder literally rather than treating
    // it as a template). Ollama's output shape is guaranteed by the JSON
    // schema passed via `format`, so its prompt only needs to describe what
    // goes in each field, not how to format the response.
    let prompt = if use_gemini {
        let format_example = match &example_snippet {
            Some((path, line)) => format!(
                "Example (format only — base the SEARCH text on lines that actually \
                appear in the codebase context below, not on this example):\n\
                FILE: {}\n\
                <<<<<<< SEARCH\n\
                {}\n\
                =======\n\
                {}\n\
                >>>>>>> REPLACE\n",
                path, line, line
            ),
            None => String::from(
                "Example (format only):\n\
                FILE: path/to/file.ext\n\
                <<<<<<< SEARCH\n\
                exact code lines to match and replace\n\
                =======\n\
                new code lines to insert\n\
                >>>>>>> REPLACE\n",
            ),
        };

        format!(
            "You are an AI coding agent modifying source code.\n\
            Perform the requested edit strictly using SEARCH/REPLACE blocks formatted exactly as follows:\n\n\
            {}\n\
            Rules:\n\
            1. Keep SEARCH blocks small and unique so they match accurately.\n\
            2. Preserve exact indentation.\n\
            3. The SEARCH text must be copied verbatim from the CODEBASE CONTEXT below — never invent or paraphrase it.\n\
            4. Do not output conversational text or markdown code fences outside of the block structure.\n\n\
            CODEBASE CONTEXT:\n{}\n\n\
            INSTRUCTION:\n{}\n",
            format_example, context_payload, instruction
        )
    } else {
        format!(
            "You are an AI coding agent modifying source code.\n\
            For each change needed, provide:\n\
            - file_path: the exact path as it appears in CODEBASE CONTEXT below\n\
            - search: text copied verbatim from that file's content — never invent, paraphrase, \
              or reformat it. It must match the file exactly so it can be located and replaced.\n\
            - replace: the new text that should take its place\n\n\
            Keep each search value small and unique enough to match exactly once. Preserve exact indentation.\n\n\
            IMPORTANT: search can never be empty. If you are ADDING new code rather than changing \
            existing code, pick a short existing line near where the addition belongs (e.g. the last \
            line of a function, or a closing brace) as search, and set replace to that same line \
            followed by the new code on the following lines.\n\n\
            CODEBASE CONTEXT:\n{}\n\n\
            INSTRUCTION:\n{}\n",
            context_payload, instruction
        )
    };

    let response_text = if use_gemini {
        let api_key = std::env::var("GEMINI_API_KEY")
            .map_err(|_| "GEMINI_API_KEY environment variable not set.")?;
        let model_name = model.trim_start_matches("models/");

        println!(
            "\n{}",
            format!(" Generating diffs using Gemini ({}) ...", model_name)
                .bold()
                .cyan()
        );

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model_name, api_key
        );

        let body = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart { text: prompt }],
            }],
        };

        let res = client.post(&url).json(&body).send().await?;
        if res.status().is_success() {
            let gemini_res: GeminiResponse = res.json().await?;
            gemini_res
                .candidates
                .and_then(|c| c.first().cloned())
                .and_then(|c| c.content.parts.first().cloned())
                .map(|p| p.text)
                .unwrap_or_default()
        } else {
            return Err(format!("Gemini API Error: {}", res.text().await?).into());
        }
    } else {
        println!(
            "\n{}",
            format!(" Generating diffs using local Ollama ({}) ...", model)
                .bold()
                .magenta()
        );

        let res = client
            .post("http://localhost:11434/api/generate")
            .json(&OllamaRequest {
                model: model.to_string(),
                prompt,
                stream: false,
                // Constrain decoding to the edit-block schema so the model
                // physically cannot emit prose, markdown fences, or a
                // half-finished block — only valid JSON matching the shape.
                format: Some(edit_blocks_schema()),
            })
            .send()
            .await?;

        let ollama_res: OllamaResponse = res.json().await?;
        ollama_res.response
    };

    // Gemini's response is still marker-format text; Ollama's is
    // schema-constrained JSON. Parse each with the matching parser.
    let edit_blocks = if use_gemini {
        parse_edit_blocks(&response_text)
    } else {
        parse_edit_blocks_json(&response_text)
    };

    if edit_blocks.is_empty() {
        println!(
            "{}",
            "No valid edit blocks were generated by the AI.".yellow()
        );
        println!("\n{}", "--- Raw model output (for debugging) ---".dimmed());
        println!("{}", response_text);
        return Ok(());
    }

    // Pre-validate every block against the actual file on disk *before*
    // showing anything to the user, so we never present a diff that can't
    // possibly be applied (missing file, or SEARCH text that doesn't match).
    let mut valid_blocks: Vec<(EditBlock, std::path::PathBuf)> = Vec::new();

    for block in edit_blocks {
        let resolved = match resolve_file_path(&block.file_path, target_dir) {
            Some(p) => p,
            None => {
                eprintln!(
                    "{}",
                    format!(
                        " Skipping block: file not found: {}",
                        block.file_path
                    )
                    .yellow()
                );
                continue;
            }
        };

        let content = match fs::read_to_string(&resolved) {
            Ok(c) => c,
            Err(_) => {
                eprintln!(
                    "{}",
                    format!(" Skipping block: could not read {}", resolved.display()).yellow()
                );
                continue;
            }
        };

        let normalized_content = content.replace("\r\n", "\n");
        let normalized_search = block.search.replace("\r\n", "\n");

        if !normalized_content.contains(&normalized_search) {
            eprintln!(
                "{}",
                format!(
                    " Skipping block: SEARCH text not found in {}",
                    resolved.display()
                )
                .yellow()
            );
            continue;
        }

        valid_blocks.push((block, resolved));
    }

    if valid_blocks.is_empty() {
        println!(
            "{}",
            "No valid, applicable edit blocks were generated by the AI.".yellow()
        );
        println!("\n{}", "--- Raw model output (for debugging) ---".dimmed());
        println!("{}", response_text);
        return Ok(());
    }

    println!("\n{}", "Proposed Modifications:".bold().magenta());
    println!(
        "{}",
        "==================================================".dimmed()
    );

    for (block, resolved) in &valid_blocks {
        println!("\nFile: {}", resolved.display().to_string().bold().green());
        for line in block.search.lines() {
            println!("  {}", format!("- {}", line).red());
        }
        for line in block.replace.lines() {
            println!("  {}", format!("+ {}", line).green());
        }
    }

    println!(
        "\n{}",
        "==================================================".dimmed()
    );
    print!("Apply these changes to disk? [y/N]: ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim().eq_ignore_ascii_case("y") {
        let blocks_only: Vec<EditBlock> = valid_blocks.into_iter().map(|(b, _)| b).collect();
        let applied_count = apply_edit_blocks(&blocks_only, target_dir)?;
        if applied_count > 0 {
            println!(
                "{}",
                format!(
                    " Successfully applied {} file modification(s)!",
                    applied_count
                )
                .green()
                .bold()
            );
        } else {
            println!(
                "{}",
                " No modifications were applied due to missing files or search mismatches."
                    .red()
                    .bold()
            );
        }
    } else {
        println!("{}", "Operation canceled. No files were modified.".yellow());
    }

    Ok(())
}

/// Parses marker-format SEARCH/REPLACE blocks from a text response (Gemini path).
fn parse_edit_blocks(raw_text: &str) -> Vec<EditBlock> {
    let mut blocks = Vec::new();
    let mut current_file = String::new();
    let mut in_search = false;
    let mut in_replace = false;
    let mut search_lines = Vec::new();
    let mut replace_lines = Vec::new();

    for line in raw_text.lines() {
        if line.starts_with("FILE:") {
            current_file = line.trim_start_matches("FILE:").trim().to_string();
        } else if line.starts_with("<<<<<<< SEARCH") {
            in_search = true;
            search_lines.clear();
        } else if line.starts_with("=======") {
            in_search = false;
            in_replace = true;
            replace_lines.clear();
        } else if line.starts_with(">>>>>>> REPLACE") {
            in_replace = false;
            // Guard against emitting garbage blocks: skip anything without a
            // real target file or an empty SEARCH section (both indicate the
            // model didn't ground its output in the actual context).
            if !current_file.is_empty() && !search_lines.is_empty() {
                blocks.push(EditBlock {
                    file_path: current_file.clone(),
                    search: search_lines.join("\n"),
                    replace: replace_lines.join("\n"),
                });
            }
        } else if in_search {
            search_lines.push(line);
        } else if in_replace {
            replace_lines.push(line);
        }
    }

    blocks
}

/// JSON schema used to constrain Ollama's decoding for `cix edit` so the
/// response can only ever be a list of `{file_path, search, replace}` objects
/// — no prose, no markdown fences, no half-finished blocks.
fn edit_blocks_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": { "type": "string" },
                        "search": { "type": "string" },
                        "replace": { "type": "string" }
                    },
                    "required": ["file_path", "search", "replace"]
                }
            }
        },
        "required": ["edits"]
    })
}

#[derive(Deserialize)]
struct EditBlocksResponse {
    edits: Vec<EditBlockJson>,
}

#[derive(Deserialize)]
struct EditBlockJson {
    file_path: String,
    search: String,
    replace: String,
}

/// Parses schema-constrained JSON edit blocks from an Ollama response.
fn parse_edit_blocks_json(raw_text: &str) -> Vec<EditBlock> {
    match serde_json::from_str::<EditBlocksResponse>(raw_text) {
        Ok(resp) => resp
            .edits
            .into_iter()
            .filter(|e| {
                if e.file_path.is_empty() || e.search.is_empty() {
                    eprintln!(
                        "{}",
                        format!(
                            " Dropping block with empty file_path/search for {}",
                            if e.file_path.is_empty() { "<empty>" } else { &e.file_path }
                        )
                        .yellow()
                    );
                    return false;
                }
                true
            })
            .map(|e| EditBlock {
                file_path: e.file_path,
                search: e.search,
                replace: e.replace,
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Resolves candidate file paths proposed by the AI to actual paths on disk.
fn resolve_file_path(proposed_path: &str, target_dir: &str) -> Option<std::path::PathBuf> {
    let clean_proposed = proposed_path.trim_start_matches(r"\\?\").replace('\\', "/");
    let path = std::path::Path::new(&clean_proposed);

    if path.exists() {
        return Some(path.to_path_buf());
    }

    let target_base = std::path::Path::new(target_dir);

    let joined = target_base.join(&clean_proposed);
    if joined.exists() {
        return Some(joined);
    }

    if let Some(target_folder_name) = target_base.file_name().and_then(|n| n.to_str()) {
        let prefix_to_strip = format!("{}/", target_folder_name);
        if clean_proposed.starts_with(&prefix_to_strip) {
            let stripped = &clean_proposed[prefix_to_strip.len()..];
            let joined_stripped = target_base.join(stripped);
            if joined_stripped.exists() {
                return Some(joined_stripped);
            }
        }
    }

    if let Some(file_name) = path.file_name() {
        for entry in ignore::WalkBuilder::new(target_base).build().flatten() {
            if entry.file_type().map_or(false, |ft| ft.is_file()) {
                if entry.file_name() == file_name {
                    return Some(entry.path().to_path_buf());
                }
            }
        }
    }

    None
}

/// Applies a list of `EditBlock` modifications to files on disk, handling line ending normalization.
fn apply_edit_blocks(
    blocks: &[EditBlock],
    target_dir: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut applied_count = 0;

    for block in blocks {
        let file_path = match resolve_file_path(&block.file_path, target_dir) {
            Some(p) => p,
            None => {
                eprintln!(
                    "{}",
                    format!(" Warning: File not found: {}", block.file_path).yellow()
                );
                continue;
            }
        };

        let content = fs::read_to_string(&file_path)?;

        let normalized_content = content.replace("\r\n", "\n");
        let normalized_search = block.search.replace("\r\n", "\n");
        let normalized_replace = block.replace.replace("\r\n", "\n");

        if !normalized_content.contains(&normalized_search) {
            eprintln!(
                "{}",
                format!(
                    " Search block missing or mismatched in {}",
                    file_path.display()
                )
                .red()
            );
            continue;
        }

        let new_content = normalized_content.replacen(&normalized_search, &normalized_replace, 1);

        let final_content = if content.contains("\r\n") {
            new_content.replace('\n', "\r\n")
        } else {
            new_content
        };

        fs::write(&file_path, final_content)?;
        applied_count += 1;
    }

    Ok(applied_count)
}

/// Opens or builds the Tantivy search index for the target directory incrementally.
fn ensure_index(
    target_directory: &str,
    reindex: bool,
    app_cache_dir: &std::path::Path,
) -> Result<(Index, Field, Field), Box<dyn std::error::Error>> {
    let target_path = fs::canonicalize(target_directory)
        .unwrap_or_else(|_| std::path::PathBuf::from(target_directory));

    let mut hasher = DefaultHasher::new();
    target_path.hash(&mut hasher);
    let dir_hash = hasher.finish();

    let index_path = app_cache_dir.join(format!("{}", dir_hash));

    if reindex && index_path.exists() {
        let _ = fs::remove_dir_all(&index_path);
    }

    fs::create_dir_all(&index_path)?;

    let state_file = index_path.join(".last_run");
    let last_run: u64 = if reindex {
        0
    } else {
        fs::read_to_string(&state_file)
            .unwrap_or_else(|_| "0".to_string())
            .trim()
            .parse()
            .unwrap_or(0)
    };

    let current_run = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    let mut schema_builder = Schema::builder();
    // TEXT field so file paths are tokenized and searchable by file name
    let file_path = schema_builder.add_text_field("path", TEXT | STORED);
    let content = schema_builder.add_text_field("content", TEXT | STORED);
    let schema = schema_builder.build();

    let dir = tantivy::directory::MmapDirectory::open(&index_path)?;
    let index = Index::open_or_create(dir, schema)?;
    let mut index_writer = index.writer(100_000_000)?;

    for result in WalkBuilder::new(&target_path).hidden(true).build() {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };

        if entry.file_type().map_or(false, |ft| ft.is_file()) {
            let raw_path_str = entry.path().to_string_lossy().to_string();
            let path_str = raw_path_str.trim_start_matches(r"\\?\").to_string();

            if is_indexable_file(&path_str) {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified_time) = metadata.modified() {
                        let modified_secs = modified_time.duration_since(UNIX_EPOCH)?.as_secs();

                        if modified_secs >= last_run {
                            // Lossy reading to support files with non-UTF8 encodings
                            if let Ok(bytes) = fs::read(entry.path()) {
                                let file_content = String::from_utf8_lossy(&bytes).to_string();
                                let path_term = Term::from_field_text(file_path, &path_str);
                                index_writer.delete_term(path_term);
                                index_writer.add_document(doc!(
                                    file_path => path_str,
                                    content => file_content
                                ))?;
                            }
                        }
                    }
                }
            }
        }
    }

    index_writer.commit()?;
    fs::write(state_file, current_run.to_string())?;

    Ok((index, file_path, content))
}

/// Checks whether a given file path should be indexed based on file extension and path exclusions.
fn is_indexable_file(path_str: &str) -> bool {
    let path_lower = path_str.to_lowercase();

    if path_lower.contains("textmesh pro")
        || path_lower.contains("node_modules")
        || path_lower.contains("target")
        || path_lower.contains("/.git/")
        || path_lower.contains("\\.git\\")
    {
        return false;
    }

    let allowed_extensions = [
        "rs", "c", "cpp", "h", "hpp", "py", "js", "ts", "toml", "json", "yaml", "yml", "md", "sh",
        "cs", "shader", "hlsl",
    ];

    std::path::Path::new(path_str)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| allowed_extensions.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Constructs a Tantivy query from user input, applying fuzzy search terms (~1)
/// to keywords so search handles typos and approximate matches.
fn parse_query_with_fuzzy(
    query_parser: &QueryParser,
    input_text: &str,
) -> Box<dyn tantivy::query::Query> {
    let keywords = extract_keywords(input_text);
    if keywords.trim().is_empty() {
        return query_parser
            .parse_query("*")
            .unwrap_or_else(|_| query_parser.parse_query("").unwrap());
    }

    let fuzzy_terms: String = keywords
        .split_whitespace()
        .map(|w| {
            if w.contains('~') || w.contains('*') || w.contains(':') || w.contains('"') {
                w.to_string()
            } else if w.len() > 2 {
                format!("{}~1", w)
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    let combined_query_str = if fuzzy_terms != keywords {
        format!("({}) OR ({})", keywords, fuzzy_terms)
    } else {
        keywords.clone()
    };

    if let Ok(q) = query_parser.parse_query(&combined_query_str) {
        return q;
    }

    if let Ok(q) = query_parser.parse_query(&keywords) {
        return q;
    }

    if let Ok(q) = query_parser.parse_query(&fuzzy_terms) {
        return q;
    }

    query_parser.parse_query("*").unwrap()
}

/// Checks if a line matches a keyword either as a substring or via fuzzy matching.
fn line_matches_keyword(line_lower: &str, kw: &str) -> bool {
    if line_lower.contains(kw) {
        return true;
    }
    let words = line_lower.split(|c: char| !c.is_alphanumeric() && c != '_');
    for word in words {
        if !word.is_empty() && is_fuzzy_match(kw, word) {
            return true;
        }
    }
    false
}

/// Determines if two words match fuzzily based on Levenshtein distance.
fn is_fuzzy_match(keyword: &str, word: &str) -> bool {
    let kw_len = keyword.len();
    let w_len = word.len();

    if kw_len == 0 || w_len == 0 {
        return false;
    }

    if word.contains(keyword) || keyword.contains(word) {
        return true;
    }

    let max_dist = if kw_len <= 3 { 0 } else if kw_len <= 6 { 1 } else { 2 };
    let len_diff = if kw_len > w_len {
        kw_len - w_len
    } else {
        w_len - kw_len
    };

    if len_diff > max_dist {
        return false;
    }

    levenshtein_distance(keyword, word) <= max_dist
}

/// Computes the Levenshtein distance between two string slices.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let len_a = a_chars.len();
    let len_b = b_chars.len();

    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }

    let mut dp = vec![vec![0; len_b + 1]; len_a + 1];

    for i in 0..=len_a {
        dp[i][0] = i;
    }
    for j in 0..=len_b {
        dp[0][j] = j;
    }

    for i in 1..=len_a {
        for j in 1..=len_b {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[len_a][len_b]
}

/// Highlights search query keyword matches (including fuzzy matches) in a code line using ANSI color formatting.
fn highlight_match(line: &str, query: &str) -> String {
    let keywords: Vec<String> = extract_keywords(query)
        .split_whitespace()
        .map(|s| s.to_lowercase())
        .collect();

    if keywords.is_empty() {
        return line.to_string();
    }

    let mut result = String::new();
    let mut current_token = String::new();
    let mut is_alphanumeric_mode = false;

    for c in line.chars() {
        let is_alphanumeric = c.is_alphanumeric() || c == '_';
        if is_alphanumeric == is_alphanumeric_mode {
            current_token.push(c);
        } else {
            if !current_token.is_empty() {
                if is_alphanumeric_mode {
                    let token_lower = current_token.to_lowercase();
                    if keywords.iter().any(|kw| line_matches_keyword(&token_lower, kw)) {
                        result.push_str(&current_token.bold().red().to_string());
                    } else {
                        result.push_str(&current_token);
                    }
                } else {
                    result.push_str(&current_token);
                }
            }
            current_token = String::new();
            current_token.push(c);
            is_alphanumeric_mode = is_alphanumeric;
        }
    }

    if !current_token.is_empty() {
        if is_alphanumeric_mode {
            let token_lower = current_token.to_lowercase();
            if keywords.iter().any(|kw| line_matches_keyword(&token_lower, kw)) {
                result.push_str(&current_token.bold().red().to_string());
            } else {
                result.push_str(&current_token);
            }
        } else {
            result.push_str(&current_token);
        }
    }

    result
}
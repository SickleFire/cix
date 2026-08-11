//! # cix - Indexed Code Search & RAG Tool
//!
//! `cix` is a CLI tool providing fast indexed code search, codebase RAG question-answering (`cix ask`),
//! and AI-driven automated code modification (`cix edit`).

use clap::{Parser, Subcommand};
use colored::*;
use futures_util::StreamExt;
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use cix::chunking::{chunk_code, CodeChunk};
use std::collections::hash_map::DefaultHasher;
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

/// Max total characters of code context assembled into a single RAG/edit
/// prompt. Rough proxy for token budget (roughly 3-4 chars/token for code),
/// chosen to leave headroom under typical local-model context windows once
/// the instruction, formatting, and system prompt text are added on top.
const CONTEXT_CHAR_BUDGET: usize = 12_000;

/// How many candidate chunks to pull from Tantivy before truncating by
/// budget. Kept generous since Tantivy search itself is cheap — the real
/// cost is prompt size, which the budget below controls directly.
const CANDIDATE_DOC_LIMIT: usize = 20;

/// Selects a prefix of `chunks` (already ranked by relevance/merge order)
/// whose combined formatted size stays within `budget` characters. Always
/// includes at least the first chunk, even if it alone exceeds the budget,
/// so a single large but highly relevant match is never dropped to zero
/// context.
fn take_chunks_within_budget(chunks: Vec<ContextChunk>, budget: usize) -> Vec<ContextChunk> {
    let mut selected = Vec::new();
    let mut running_len = 0usize;

    for (i, chunk) in chunks.into_iter().enumerate() {
        // Mirrors the "--- FILE: {} (lines {}-{}) ---\n" formatting overhead
        // added when building context_payload, so the budget reflects what
        // actually lands in the prompt, not just raw chunk.content length.
        let formatted_len = chunk.content.len() + chunk.path.len() + 40;

        if i == 0 {
            selected.push(chunk);
            running_len += formatted_len;
            continue;
        }

        if running_len + formatted_len > budget {
            break;
        }

        running_len += formatted_len;
        selected.push(chunk);
    }

    selected
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

    let (index, file_path_field, content_field, start_line_field, end_line_field) =
        ensure_index(target_dir, false, app_cache_dir)?;
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    // Search across both file path and content fields
    let query_parser = QueryParser::for_index(&index, vec![file_path_field, content_field]);
    let query = parse_query_with_fuzzy(&query_parser, question);

    let mut top_docs = searcher.search(&query, &TopDocs::with_limit(CANDIDATE_DOC_LIMIT).and_offset(0))?;

    // Fallback: If generic question returns no hits, pull top indexed files
    if top_docs.is_empty() {
        if let Ok(fallback_query) = query_parser.parse_query("*") {
            top_docs = searcher.search(&fallback_query, &TopDocs::with_limit(CANDIDATE_DOC_LIMIT).and_offset(0))?;
        }
    }

    if top_docs.is_empty() {
        println!("{}", "No relevant code files found for context.".yellow());
        return Ok(());
    }

    let mut context_payload = String::new();
    println!("{}", "Found relevant context in:".dimmed());

    let merged_chunks = build_merged_context_chunks(
        &top_docs,
        &searcher,
        file_path_field,
        content_field,
        start_line_field,
        end_line_field,
    )?;

    let merged_chunks = take_chunks_within_budget(merged_chunks, CONTEXT_CHAR_BUDGET);

    for chunk in &merged_chunks {
        println!(
            "  • {} (lines {}-{})",
            chunk.path.bold().green(),
            chunk.start_line,
            chunk.end_line
        );
        context_payload.push_str(&format!(
            "\n--- FILE: {} (lines {}-{}) ---\n{}\n",
            chunk.path, chunk.start_line, chunk.end_line, chunk.content
        ));
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
    let (index, file_path, content, _start_line, _end_line) =
        ensure_index(target_directory, reindex, app_cache_dir)?;

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
                if keywords
                    .iter()
                    .any(|kw| line_matches_keyword(&line_lower, kw))
                {
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
    mode: EditMode,
    search: String,
    replace: String,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum EditMode {
    Replace,
    Append,
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

    let (index, file_path_field, content_field, start_line_field, end_line_field) =
        ensure_index(target_dir, false, app_cache_dir)?;
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();

    let query_parser = QueryParser::for_index(&index, vec![file_path_field, content_field]);
    let query = parse_query_with_fuzzy(&query_parser, instruction);

    let mut top_docs = searcher.search(&query, &TopDocs::with_limit(CANDIDATE_DOC_LIMIT).and_offset(0))?;

    if top_docs.is_empty() {
        if let Ok(fallback_query) = query_parser.parse_query("*") {
            top_docs = searcher.search(&fallback_query, &TopDocs::with_limit(CANDIDATE_DOC_LIMIT).and_offset(0))?;
        }
    }

    if top_docs.is_empty() {
        println!("{}", "No relevant code files found to modify.".yellow());
        return Ok(());
    }

    let mut context_payload = String::new();
    let mut example_snippet: Option<(String, String)> = None; // (path, first_line) for the few-shot example
    println!("{}", "Target files for context:".dimmed());

    let merged_chunks = build_merged_context_chunks(
        &top_docs,
        &searcher,
        file_path_field,
        content_field,
        start_line_field,
        end_line_field,
    )?;

    let merged_chunks = take_chunks_within_budget(merged_chunks, CONTEXT_CHAR_BUDGET);

    for chunk in &merged_chunks {
        if example_snippet.is_none() {
            if let Some(first_line) = chunk.content.lines().find(|l| !l.trim().is_empty()) {
                example_snippet = Some((chunk.path.clone(), first_line.to_string()));
            }
        }

        println!(
            "  • {} (lines {}-{})",
            chunk.path.bold().green(),
            chunk.start_line,
            chunk.end_line
        );
        context_payload.push_str(&format!(
            "\n--- FILE: {} (lines {}-{}) ---\n{}\n",
            chunk.path, chunk.start_line, chunk.end_line, chunk.content
        ));
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
        let anchor_example = match &example_snippet {
            Some((path, line)) => format!(
                "\nExample of adding NEW content at the end of a file:\n\
                {{\"file_path\": \"{}\", \"mode\": \"append\", \"search\": \"\", \"replace\": \"<new content goes here>\"}}\n\
                (append mode adds `replace` to the end of the file — no anchor needed)\n\n\
                Example of changing EXISTING content (format only — base the real search on a line \
                that actually appears in the codebase context below, not on this example):\n\
                {{\"file_path\": \"{}\", \"mode\": \"replace\", \"search\": \"{}\", \"replace\": \"<updated line>\"}}\n",
                path, path, line
            ),
            None => String::new(),
        };

        format!(
            "You are an AI coding agent modifying source code.\n\
            For each change needed, provide:\n\
            - file_path: the exact path as it appears in CODEBASE CONTEXT below\n\
            - mode: either \"replace\" (change existing text) or \"append\" (add new text to the end of the file)\n\
            - search: for mode \"replace\", text copied verbatim from that file's content — never invent, \
              paraphrase, or reformat it. It must match the file exactly so it can be located and replaced. \
              For mode \"append\", leave this as an empty string.\n\
            - replace: for mode \"replace\", the new text that should take the place of search. \
              For mode \"append\", the new content to add at the end of the file.\n\n\
            Keep each search value small and unique enough to match exactly once. Preserve exact indentation.\n\n\
            IMPORTANT: If the instruction asks to ADD new content (e.g. a new section, a new line, \
            \"at the bottom\", \"at the end\"), use mode \"append\" rather than trying to force it \
            through a replace block.\n{}\n\
            CODEBASE CONTEXT:\n{}\n\n\
            INSTRUCTION:\n{}\n",
            anchor_example, context_payload, instruction
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

    // Gemini's response is still marker-format text (replace-only); Ollama's
    // is schema-constrained JSON (replace + append). Parse each with the
    // matching parser.
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
                    format!(" Skipping block: file not found: {}", block.file_path).yellow()
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

        // Append blocks don't need SEARCH text to exist anywhere — they just
        // need a readable target file, which we already confirmed above.
        if block.mode == EditMode::Replace {
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
        match block.mode {
            EditMode::Replace => {
                for line in block.search.lines() {
                    println!("  {}", format!("- {}", line).red());
                }
                for line in block.replace.lines() {
                    println!("  {}", format!("+ {}", line).green());
                }
            }
            EditMode::Append => {
                println!("  {}", "(appending to end of file)".dimmed());
                for line in block.replace.lines() {
                    println!("  {}", format!("+ {}", line).green());
                }
            }
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
/// Gemini's prompt only teaches the replace format, so every parsed block is
/// tagged `EditMode::Replace`.
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
                    mode: EditMode::Replace,
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
/// response can only ever be a list of `{file_path, mode, search, replace}`
/// objects — no prose, no markdown fences, no half-finished blocks.
fn edit_blocks_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": { "type": "string", "minLength": 1 },
                        "mode": { "type": "string", "enum": ["replace", "append"] },
                        "search": { "type": "string" },
                        "replace": { "type": "string", "minLength": 1 }
                    },
                    "required": ["file_path", "mode", "replace"]
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
    mode: String,
    #[serde(default)]
    search: String,
    replace: String,
}

/// Parses schema-constrained JSON edit blocks from an Ollama response.
fn parse_edit_blocks_json(raw_text: &str) -> Vec<EditBlock> {
    match serde_json::from_str::<EditBlocksResponse>(raw_text) {
        Ok(resp) => resp
            .edits
            .into_iter()
            .filter_map(|e| {
                if e.file_path.is_empty() {
                    eprintln!("{}", " Dropping block with empty file_path".yellow());
                    return None;
                }
                if e.replace.is_empty() {
                    eprintln!(
                        "{}",
                        format!(" Dropping block with empty replace for {}", e.file_path).yellow()
                    );
                    return None;
                }

                let mode = match e.mode.as_str() {
                    "append" => EditMode::Append,
                    "replace" => EditMode::Replace,
                    other => {
                        eprintln!(
                            "{}",
                            format!(
                                " Unrecognized mode '{}' for {}, treating as replace",
                                other, e.file_path
                            )
                            .yellow()
                        );
                        EditMode::Replace
                    }
                };

                if mode == EditMode::Replace && e.search.is_empty() {
                    eprintln!(
                        "{}",
                        format!(
                            " Dropping replace block with empty search for {}",
                            e.file_path
                        )
                        .yellow()
                    );
                    return None;
                }

                Some(EditBlock {
                    file_path: e.file_path,
                    mode,
                    search: e.search,
                    replace: e.replace,
                })
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

/// Applies a list of `EditBlock` modifications to files on disk, handling line
/// ending normalization. `Replace` blocks find-and-replace a verbatim SEARCH
/// match; `Append` blocks add `replace` to the end of the file unconditionally.
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

        match block.mode {
            EditMode::Replace => {
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

                let new_content =
                    normalized_content.replacen(&normalized_search, &normalized_replace, 1);

                let final_content = if content.contains("\r\n") {
                    new_content.replace('\n', "\r\n")
                } else {
                    new_content
                };

                fs::write(&file_path, final_content)?;
                applied_count += 1;
            }
            EditMode::Append => {
                let normalized_replace = block.replace.replace("\r\n", "\n");
                let separator = if content.is_empty() || content.ends_with('\n') {
                    ""
                } else {
                    "\n"
                };
                let new_content = format!(
                    "{}{}{}\n",
                    content.replace("\r\n", "\n"),
                    separator,
                    normalized_replace
                );

                let final_content = if content.contains("\r\n") {
                    new_content.replace('\n', "\r\n")
                } else {
                    new_content
                };

                fs::write(&file_path, final_content)?;
                applied_count += 1;
            }
        }
    }

    Ok(applied_count)
}

#[derive(Debug, Clone, PartialEq)]
struct ContextChunk {
    path: String,
    start_line: u64,
    end_line: u64,
    content: String,
}

/// Builds merged context chunks from retrieved search documents by grouping by file_path,
/// sorting by start_line, and merging contiguous or overlapping line ranges.
fn build_merged_context_chunks(
    top_docs: &[(tantivy::Score, tantivy::DocAddress)],
    searcher: &tantivy::Searcher,
    file_path_field: Field,
    content_field: Field,
    start_line_field: Field,
    end_line_field: Field,
) -> Result<Vec<ContextChunk>, Box<dyn std::error::Error>> {
    let mut file_order = Vec::new();
    let mut file_chunks: std::collections::HashMap<String, Vec<ContextChunk>> =
        std::collections::HashMap::new();

    for (_, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(*doc_address)?;
        let path = retrieved_doc
            .get_first(file_path_field)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();

        let start = retrieved_doc
            .get_first(start_line_field)
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        let end = retrieved_doc
            .get_first(end_line_field)
            .and_then(|v| v.as_u64())
            .unwrap_or(1);

        let content = retrieved_doc
            .get_first(content_field)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();

        if !file_chunks.contains_key(&path) {
            file_order.push(path.clone());
        }
        file_chunks.entry(path.clone()).or_default().push(ContextChunk {
            path,
            start_line: start,
            end_line: end,
            content,
        });
    }

    let mut merged_chunks = Vec::new();
    for path in file_order {
        if let Some(chunks) = file_chunks.remove(&path) {
            merged_chunks.extend(merge_chunks(chunks));
        }
    }

    Ok(merged_chunks)
}

/// Sorts chunks by start_line and merges contiguous or overlapping line ranges for a single file.
fn merge_chunks(mut chunks: Vec<ContextChunk>) -> Vec<ContextChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    chunks.sort_by(|a, b| {
        a.start_line
            .cmp(&b.start_line)
            .then_with(|| a.end_line.cmp(&b.end_line))
    });

    let mut merged: Vec<ContextChunk> = Vec::new();
    for chunk in chunks {
        if merged.is_empty() {
            merged.push(chunk);
        } else {
            let last = merged.last_mut().unwrap();
            if chunk.start_line <= last.end_line + 1 {
                if chunk.end_line > last.end_line {
                    let overlap = if last.end_line >= chunk.start_line {
                        (last.end_line - chunk.start_line + 1) as usize
                    } else {
                        0
                    };
                    let new_lines: Vec<&str> = chunk.content.lines().skip(overlap).collect();
                    if !new_lines.is_empty() {
                        if !last.content.is_empty() {
                            last.content.push('\n');
                        }
                        last.content.push_str(&new_lines.join("\n"));
                    }
                    last.end_line = chunk.end_line;
                }
            } else {
                merged.push(chunk);
            }
        }
    }
    merged
}

/// Opens or builds the Tantivy search index for the target directory incrementally.
fn ensure_index(
    target_directory: &str,
    reindex: bool,
    app_cache_dir: &std::path::Path,
) -> Result<(Index, Field, Field, Field, Field), Box<dyn std::error::Error>> {
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
    let start_line = schema_builder.add_u64_field("start_line", INDEXED | STORED);
    let end_line = schema_builder.add_u64_field("end_line", INDEXED | STORED);
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

                                let ext = std::path::Path::new(&path_str)
                                    .extension()
                                    .and_then(|e| e.to_str())
                                    .unwrap_or("");

                                let chunks = chunk_code(&file_content, ext);
                                for chunk in chunks {
                                    index_writer.add_document(doc!(
                                        file_path => path_str.clone(),
                                        content => chunk.content,
                                        start_line => chunk.start_line as u64,
                                        end_line => chunk.end_line as u64
                                    ))?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    index_writer.commit()?;
    fs::write(state_file, current_run.to_string())?;

    Ok((index, file_path, content, start_line, end_line))
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

    let max_dist = if kw_len <= 3 {
        0
    } else if kw_len <= 6 {
        1
    } else {
        2
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_keywords() {
        let question = "how does extract_keywords work in main.rs?";
        let keywords = extract_keywords(question);
        assert!(keywords.contains("extract_keywords"));
        assert!(!keywords.contains("how"));
        assert!(!keywords.contains("does"));
        assert!(!keywords.contains("main.rs"));
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("flaw", "lawn"), 2);
        assert_eq!(levenshtein_distance("test", "test"), 0);
    }

    #[test]
    fn test_is_fuzzy_match() {
        assert!(is_fuzzy_match("search", "search"));
        assert!(is_fuzzy_match("search", "seach")); // 1 typo
        assert!(!is_fuzzy_match("search", "completelydifferent"));
    }

    #[test]
    fn test_is_indexable_file() {
        assert!(is_indexable_file("src/main.rs"));
        assert!(is_indexable_file("script.py"));
        assert!(!is_indexable_file("node_modules/package/index.js"));
        assert!(!is_indexable_file(".git/config"));
        assert!(!is_indexable_file("archive.zip"));
    }

    #[test]
    fn test_parse_edit_blocks() {
        let raw =
            "FILE: src/test.rs\n<<<<<<< SEARCH\nold line\n=======\nnew line\n>>>>>>> REPLACE\n";
        let blocks = parse_edit_blocks(raw);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].file_path, "src/test.rs");
        assert_eq!(blocks[0].search, "old line");
        assert_eq!(blocks[0].replace, "new line");
        assert_eq!(blocks[0].mode, EditMode::Replace);
    }

    #[test]
    fn test_chunk_code() {
        let short_code = "fn main() {\n    println!(\"hello\");\n}";
        assert_eq!(chunk_code(short_code).len(), 1);

        let long_code = (0..40)
            .map(|i| format!("// Line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_code(&long_code);
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_merge_chunks() {
        let chunk1 = ContextChunk {
            path: "file.rs".to_string(),
            start_line: 1,
            end_line: 3,
            content: "line 1\nline 2\nline 3".to_string(),
        };
        let chunk2 = ContextChunk {
            path: "file.rs".to_string(),
            start_line: 3,
            end_line: 5,
            content: "line 3\nline 4\nline 5".to_string(),
        };
        let chunk3 = ContextChunk {
            path: "file.rs".to_string(),
            start_line: 10,
            end_line: 12,
            content: "line 10\nline 11\nline 12".to_string(),
        };

        let merged = merge_chunks(vec![chunk2, chunk1, chunk3]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].start_line, 1);
        assert_eq!(merged[0].end_line, 5);
        assert_eq!(
            merged[0].content,
            "line 1\nline 2\nline 3\nline 4\nline 5"
        );
        assert_eq!(merged[1].start_line, 10);
        assert_eq!(merged[1].end_line, 12);
    }
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
                    if keywords
                        .iter()
                        .any(|kw| line_matches_keyword(&token_lower, kw))
                    {
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
            if keywords
                .iter()
                .any(|kw| line_matches_keyword(&token_lower, kw))
            {
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

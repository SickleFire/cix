//! # cix - Indexed Code Search & RAG Tool
//!
//! `cix` is a CLI tool providing fast indexed code search, codebase RAG question-answering (`cix ask`),
//! and AI-driven automated code modification (`cix edit`).
//!
//! `ask` and `edit` are both driven by a small ReAct-style agent loop: the model is given a
//! fixed set of tools (search_codebase, read_file, list_directory, run_command, and — for
//! `edit` only — write_file_edits) and repeatedly emits a single JSON "next action" until it
//! calls `final_answer`. After edits are applied, the agent automatically runs the project's
//! build/test command (if one can be detected) and feeds any failure back to the model so it
//! can attempt a fix, up to a small retry budget.

use clap::{Parser, Subcommand};
use colored::*;
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, ReloadPolicy, TantivyDocument, Term, doc};
use tokio::process::Command as TokioCommand;
use tokio::time::timeout;

/// When true, prints the raw parsed action_input for every executed action. Useful when
/// diagnosing schema/parsing mismatches between what the model intends (per its `thought`)
/// and what actually gets executed. Cheap enough to leave in permanently.
const DEBUG_ACTIONS: bool = true;

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
    #[arg(short = 'C', long, default_value_t = 0)]
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
    /// Ask an LLM a question about your codebase using an agentic retrieval loop
    Ask {
        /// The question to ask about your codebase
        question: String,

        /// Model identifier
        #[arg(
            short = 'm',
            long,
            env = "CIX_MODEL",
            default_value = "gemini-3.5-flash-lite"
        )]
        model: String,

        /// Provider override: 'ollama', 'gemini', or 'auto'
        #[arg(short = 'p', long, default_value = "gemini")]
        provider: String,

        /// Target directory
        #[arg(default_value = ".")]
        target_directory: String,

        /// Skip the multi-step agent loop and answer using a single RAG retrieval pass
        #[arg(long, default_value_t = false)]
        one_shot: bool,
    },

    /// Request the AI agent to modify files in your codebase, verifying its own work
    Edit {
        /// Instruction for the code modification
        instruction: String,

        /// Model identifier
        #[arg(
            short = 'm',
            long,
            env = "CIX_MODEL",
            default_value = "gemini-3.5-flash-lite"
        )]
        model: String,

        /// Provider override: 'ollama', 'gemini', or 'auto'
        #[arg(short = 'p', long, default_value = "gemini")]
        provider: String,

        /// Target directory
        #[arg(default_value = ".")]
        target_directory: String,

        /// Skip agent loop and apply edits in a single RAG request
        #[arg(long, default_value_t = false)]
        one_shot: bool,
    },
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    /// JSON schema for constrained decoding. `None` for free-form answers,
    /// `Some(schema)` when the response must match a structural contract
    /// (e.g. the agent's next-action shape). Skipped entirely when absent
    /// so Ollama sees no `format` field at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "generationConfig")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    #[serde(rename = "responseMimeType")]
    response_mime_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "responseSchema")]
    response_schema: Option<serde_json::Value>,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
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

    if let Some(Commands::Ask {
        question,
        model,
        provider,
        target_directory,
        one_shot,
    }) = &cli.command
    {
        if *one_shot {
            run_one_shot_ask_pipeline(question, model, provider, target_directory, &app_cache_dir)
                .await?;
        } else {
            run_ask_agent_pipeline(question, model, provider, target_directory, &app_cache_dir)
                .await?;
        }

        return Ok(());
    }

    if let Some(Commands::Edit {
        instruction,
        model,
        provider,
        target_directory,
        one_shot,
    }) = &cli.command
    {
        if *one_shot {
            run_one_shot_edit_pipeline(
                instruction,
                model,
                provider,
                target_directory,
                &app_cache_dir,
            )
            .await?;
        } else {
            run_edit_agent_pipeline(
                instruction,
                model,
                provider,
                target_directory,
                &app_cache_dir,
            )
            .await?;
        }

        return Ok(());
    }

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

// =====================================================================================
// Agent loop: shared "next action" contract, tool implementations, and the ReAct driver
// used by both `cix ask` and `cix edit`.
// =====================================================================================

/// Raw shape of a single proposed action exactly as it comes off the wire. For Gemini,
/// per `agent_action_schema()`, `action_input` is constrained to a STRING containing
/// JSON-encoded arguments (Gemini's structured-output mode does not reliably support
/// open/arbitrary-key nested objects — an object schema with no declared `properties`
/// silently collapses to `{}` regardless of what the model intends). For Ollama, whose
/// format-constrained decoding is looser, `action_input` may already arrive as a real
/// object. This raw struct accepts either shape via `serde_json::Value` and
/// `normalize_turn` sorts out which one it actually got.
#[derive(Deserialize, Debug, Clone)]
struct ProposedActionRaw {
    action: String,
    #[serde(default)]
    action_input: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct AgentTurnRaw {
    #[serde(default)]
    thought: Option<String>,
    actions: Vec<ProposedActionRaw>,
}

/// A single "next step" the agent model proposes, normalized so `action_input` is always
/// a real JSON object/value ready for `.get(...)` regardless of whether it arrived as a
/// JSON-encoded string (Gemini) or a native object (Ollama).
#[derive(Deserialize, Debug, Clone)]
struct ProposedAction {
    action: String,
    #[serde(default)]
    action_input: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct AgentTurn {
    #[serde(default)]
    thought: Option<String>,
    actions: Vec<ProposedAction>,
}

/// Normalizes a raw parsed turn into the shape the loop uses. If `action_input` came back
/// as a JSON string (the Gemini case, forced by the schema below), parse it a second time
/// into a real value. If it's already an object (the Ollama case), pass it through as-is.
/// If the string fails to parse as JSON, it's kept as a raw string value so at least the
/// content isn't silently dropped — downstream `.get("path")` calls will just miss, and
/// the resulting tool error will show the model what it actually sent.
fn normalize_turn(raw: AgentTurnRaw) -> AgentTurn {
    let actions = raw
        .actions
        .into_iter()
        .map(|a| {
            let input = match a.action_input {
                serde_json::Value::String(s) => {
                    match serde_json::from_str(&s) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!(
                                "{}",
                                format!(
                                    "  WARNING: action_input for '{}' looked like a JSON string but failed to parse ({}). Raw: {}",
                                    a.action, e, s
                                )
                                .yellow()
                            );
                            serde_json::Value::String(s)
                        }
                    }
                }
                other => other,
            };
            ProposedAction {
                action: a.action,
                action_input: input,
            }
        })
        .collect();
    AgentTurn {
        thought: raw.thought,
        actions,
    }
}

/// JSON schema used to constrain Gemini's decoding for the agent loop so the response can
/// only ever be `{thought?, actions: [{action, action_input}, ...]}` — no prose, no
/// markdown fences, no half-finished JSON.
///
/// IMPORTANT: `action_input` is deliberately typed as a STRING (containing JSON-encoded
/// arguments), not a nested `object`. Gemini's `responseSchema` does not reliably support
/// open/arbitrary-key objects — an `object` schema with no declared `properties` tends to
/// collapse to `{}` at decode time regardless of what the model is trying to say, which
/// silently drops every argument (this was the root cause of read_file always receiving
/// an empty path). Routing arguments through a string field sidesteps that limitation;
/// `normalize_turn` parses the string back into a real value afterward.
fn agent_action_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "thought": { "type": "string" },
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_ACTIONS_PER_TURN,
                "items": {
                    "type": "object",
                    "properties": {
                        "action": { "type": "string" },
                        "action_input": {
                            "type": "string",
                            "description": "A JSON-ENCODED STRING (not a nested object) containing this action's arguments, e.g. \"{\\\"path\\\": \\\"index.html\\\", \\\"start_line\\\": 1, \\\"end_line\\\": 100}\". Must parse as valid JSON."
                        }
                    },
                    "required": ["action", "action_input"]
                }
            }
        },
        "required": ["actions"]
    })
}

/// Best-effort extraction of a JSON object from a blob of text, for models that ignore the
/// "respond with ONLY JSON" instruction and wrap it in prose or markdown fences.
fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end > start {
        Some(&text[start..=end])
    } else {
        None
    }
}

/// Calls the Gemini API once with a fully-formed prompt and returns the raw text response.
async fn call_gemini_once(
    model: &str,
    prompt: String,
    schema: Option<serde_json::Value>,
) -> Result<String, Box<dyn std::error::Error>> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| "GEMINI_API_KEY environment variable is missing.")?;
    let model_name = model.trim_start_matches("models/");

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model_name, api_key
    );

    let client = reqwest::Client::new();
    let generation_config = schema.map(|s| GeminiGenerationConfig {
        response_mime_type: "application/json".to_string(),
        response_schema: Some(s),
        // Raised from Gemini's default so a full-file generation turn (e.g. an
        // entire HTML/CSS/JS portfolio page) doesn't get cut off mid-JSON, which
        // would produce truncated/invalid output regardless of JSON mode.
        max_output_tokens: 8192,
    });
    let body = GeminiRequest {
        contents: vec![GeminiContent {
            parts: vec![GeminiPart { text: prompt }],
        }],
        generation_config,
    };

    let res = client.post(&url).json(&body).send().await?;

    if res.status().is_success() {
        let gemini_res: GeminiResponse = res.json().await?;
        Ok(gemini_res
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content.parts.into_iter().next())
            .map(|p| p.text)
            .unwrap_or_default())
    } else {
        Err(format!("Gemini API Error: {}", res.text().await?).into())
    }
}

/// Calls the local Ollama server once (non-streaming, since the agent loop needs the full
/// response text to parse a JSON action) and returns the raw text response.
async fn call_ollama_once(
    model: &str,
    prompt: String,
    format: Option<serde_json::Value>,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let res = client
        .post("http://localhost:11434/api/generate")
        .json(&OllamaRequest {
            model: model.to_string(),
            prompt,
            stream: false,
            format,
        })
        .send()
        .await
        .map_err(|_| "Failed to connect to local Ollama server at http://localhost:11434.")?;

    let ollama_res: OllamaResponse = res.json().await?;
    Ok(ollama_res.response)
}

/// Result of running a shell command via the `run_command` tool.
struct CommandOutput {
    success: bool,
    text: String,
}

/// Tool: `search_codebase(query)`. Runs a fuzzy full-text query against the Tantivy index
/// and returns a short preview of each matching file.
fn tool_search_codebase(
    query: &str,
    index: &Index,
    file_path_field: Field,
    content_field: Field,
) -> String {
    let reader = match index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()
    {
        Ok(r) => r,
        Err(e) => return format!("Error opening index reader: {}", e),
    };
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(index, vec![file_path_field, content_field]);
    let parsed_query = parse_query(&query_parser, query);

    let top_docs = match searcher.search(&parsed_query, &TopDocs::with_limit(5)) {
        Ok(d) => d,
        Err(e) => return format!("Search error: {}", e),
    };

    if top_docs.is_empty() {
        return "No results found for that query.".to_string();
    }

    let mut out = String::new();
    let mut seen = HashSet::new();
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = match searcher.doc(doc_address) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let path = retrieved_doc
            .get_first(file_path_field)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !seen.insert(path.to_string()) {
            continue;
        }
        let content = retrieved_doc
            .get_first(content_field)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let total_lines = content.lines().count();
        let preview: String = content.lines().take(15).collect::<Vec<_>>().join("\n");
        out.push_str(&format!(
            "FILE: {} (score {:.2}, {} lines total)\n{}\n---\n",
            path, score, total_lines, preview
        ));
    }
    out
}

/// Tool: `read_file(path, start_line, end_line)`. Reads a specific, bounded line range from
/// a file so the agent can pull precise context without re-reading whole files.
///
/// `raw_input` is the full `action_input` value as received (post-normalization), used only
/// to echo back exactly what the model sent when `path` is empty, so a model that used the
/// wrong field name (e.g. `file_path` instead of `path`) can see the mismatch itself instead
/// of retrying the same broken call indefinitely.
fn tool_read_file(
    target_dir: &str,
    path: &str,
    start_line: usize,
    end_line: usize,
    raw_input: &serde_json::Value,
) -> String {
    if path.trim().is_empty() {
        return format!(
            "Error: read_file requires a non-empty \"path\" (e.g. the exact path shown by \
            list_directory, such as \"index.html\"). You sent action_input: {}. The field \
            must be named \"path\" — \"file_path\" is only used by write_file_edits.",
            raw_input
        );
    }
    let resolved = match resolve_file_path(path, target_dir) {
        Some(p) => p,
        None => {
            return format!(
                "Error: file not found: '{}'. Call list_directory first and copy an exact path from its output.",
                path
            );
        }
    };

    let content = match fs::read_to_string(&resolved) {
        Ok(c) => c,
        Err(e) => return format!("Error reading {}: {}", resolved.display(), e),
    };

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    if total == 0 {
        return "(file is empty)".to_string();
    }

    let start = start_line.max(1);
    if start > total {
        return format!(
            "Error: start_line {} is beyond file length ({} lines)",
            start, total
        );
    }
    let requested_end = end_line.max(start).min(total);
    // Guard against runaway reads eating the whole context budget.
    let capped_end = requested_end.min(start + 400 - 1).min(total);

    let mut out = String::new();
    for i in start..=capped_end {
        out.push_str(&format!("{:>5}: {}\n", i, lines[i - 1]));
    }
    if capped_end < requested_end {
        out.push_str(&format!(
            "...[truncated at line {}; request a narrower range to see more]\n",
            capped_end
        ));
    }
    out
}

/// Tool: `list_directory(path)`. Lists the immediate children of a directory relative to the
/// target project root, skipping VCS/build/dependency clutter.
fn tool_list_directory(target_dir: &str, rel_path: &str) -> String {
    let base = std::path::Path::new(target_dir);
    let dir = if rel_path.trim().is_empty() || rel_path == "." {
        base.to_path_buf()
    } else {
        base.join(rel_path)
    };

    if !dir.exists() || !dir.is_dir() {
        return format!("Error: directory not found: {}", dir.display());
    }

    let mut entries = Vec::new();
    match fs::read_dir(&dir) {
        Ok(rd) => {
            for entry in rd.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('.') || name == "node_modules" || name == "target" {
                    continue;
                }
                let kind = if entry.path().is_dir() { "dir" } else { "file" };
                entries.push(format!("[{}] {}", kind, name));
            }
        }
        Err(e) => return format!("Error listing {}: {}", dir.display(), e),
    }

    entries.sort();
    if entries.is_empty() {
        "(empty directory)".to_string()
    } else {
        entries.join("\n")
    }
}

/// Tool: `run_command(cmd)`. Runs a shell command in the target project root (e.g.
/// `cargo check`, `cargo test`, `npm test`) with a timeout, and reports whether it succeeded.
async fn tool_run_command(target_dir: &str, cmd: &str) -> CommandOutput {
    println!("{}", format!(" Running: {}", cmd).cyan());

    let (shell, flag) = if cfg!(target_os = "windows") {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    };

    let fut = TokioCommand::new(shell)
        .arg(flag)
        .arg(cmd)
        .current_dir(target_dir)
        .output();

    match timeout(Duration::from_secs(120), fut).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let success = output.status.success();
            let mut combined = format!(
                "exit status: {}\nstdout:\n{}\nstderr:\n{}",
                output.status, stdout, stderr
            );
            if combined.len() > 6000 {
                combined.truncate(6000);
                combined.push_str("\n...[truncated]");
            }
            CommandOutput {
                success,
                text: combined,
            }
        }
        Ok(Err(e)) => CommandOutput {
            success: false,
            text: format!("Error executing command: {}", e),
        },
        Err(_) => CommandOutput {
            success: false,
            text: "Error: command timed out after 120 seconds".to_string(),
        },
    }
}

/// Detects a reasonable build/test command for the self-correction loop based on common
/// project marker files. Returns `None` if nothing recognizable is present, in which case
/// the agent simply isn't given automatic build feedback.
fn detect_build_command(target_dir: &str) -> Option<String> {
    let base = std::path::Path::new(target_dir);
    if base.join("Cargo.toml").exists() {
        Some("cargo check --message-format short".to_string())
    } else if base.join("package.json").exists() {
        Some("npm test --silent".to_string())
    } else if base.join("pyproject.toml").exists() || base.join("requirements.txt").exists() {
        Some("python -m pytest -q".to_string())
    } else if base.join("go.mod").exists() {
        Some("go build ./...".to_string())
    } else {
        None
    }
}

/// Parses the `edits` array from a `write_file_edits` action's `action_input` into
/// `EditBlock`s, dropping any entries missing required fields.
fn parse_edit_blocks_from_value(v: &serde_json::Value) -> Vec<EditBlock> {
    let arr = match v.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };

    arr.iter()
        .filter_map(|item| {
            let file_path = item.get("file_path")?.as_str()?.to_string();
            let mode_str = item
                .get("mode")
                .and_then(|m| m.as_str())
                .unwrap_or("replace");
            let mode = match mode_str {
                "append" => EditMode::Append,
                "create" => EditMode::Create,
                _ => EditMode::Replace,
            };
            let search = item
                .get("search")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let replace = item
                .get("replace")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();

            if replace.is_empty() {
                return None;
            }
            if mode == EditMode::Replace && search.is_empty() {
                return None;
            }

            Some(EditBlock {
                file_path,
                mode,
                search,
                replace,
            })
        })
        .collect()
}

/// Validates proposed edits against the files on disk, shows the user a diff, asks for
/// confirmation, and applies them if approved. Returns an observation string for the agent
/// plus whether anything was actually written to disk.
fn propose_and_apply_edits(edits: Vec<EditBlock>, target_dir: &str) -> (String, bool) {
    if edits.is_empty() {
        return (
            "No edit blocks were provided (or none had the required fields).".to_string(),
            false,
        );
    }

    let mut valid_blocks: Vec<(EditBlock, std::path::PathBuf)> = Vec::new();
    let mut skip_notes = Vec::new();

    for block in edits {
        let resolved = match block.mode {
            EditMode::Create => match resolve_new_file_path(&block.file_path, target_dir) {
                Some(p) => p,
                None => {
                    skip_notes.push(format!(
                        "cannot create {}: a file already exists at that path (use replace/append instead)",
                        block.file_path
                    ));
                    continue;
                }
            },
            EditMode::Replace | EditMode::Append => {
                match resolve_file_path(&block.file_path, target_dir) {
                    Some(p) => p,
                    None => {
                        skip_notes.push(format!("file not found: {}", block.file_path));
                        continue;
                    }
                }
            }
        };

        if block.mode == EditMode::Replace {
            let content = match fs::read_to_string(&resolved) {
                Ok(c) => c,
                Err(_) => {
                    skip_notes.push(format!("could not read {}", resolved.display()));
                    continue;
                }
            };
            let normalized_content = content.replace("\r\n", "\n");
            let normalized_search = block.search.replace("\r\n", "\n");
            if !normalized_content.contains(&normalized_search) {
                skip_notes.push(format!("SEARCH text not found in {}", resolved.display()));
                continue;
            }
        }

        valid_blocks.push((block, resolved));
    }

    if valid_blocks.is_empty() {
        return (
            format!(
                "No valid edit blocks could be applied. Issues: {}",
                skip_notes.join("; ")
            ),
            false,
        );
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
            EditMode::Create => {
                println!("  {}", "(creating new file)".dimmed());
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
    let _ = std::io::stdout().flush();

    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return (
            "Error reading user confirmation from stdin; edits were not applied.".to_string(),
            false,
        );
    }

    if input.trim().eq_ignore_ascii_case("y") {
        let blocks_only: Vec<EditBlock> = valid_blocks.into_iter().map(|(b, _)| b).collect();
        match apply_edit_blocks(&blocks_only, target_dir) {
            Ok(n) => (
                format!(
                    "Applied {} edit(s) to disk. Skipped: {}",
                    n,
                    if skip_notes.is_empty() {
                        "none".to_string()
                    } else {
                        skip_notes.join("; ")
                    }
                ),
                n > 0,
            ),
            Err(e) => (format!("Error applying edits: {}", e), false),
        }
    } else {
        ("User declined to apply these edits.".to_string(), false)
    }
}

/// Builds the human-readable tool listing injected into the agent's system prompt.
fn build_tools_description(allow_write: bool) -> String {
    let mut desc = String::from(
        "Available actions (each \"action\" field must be exactly one of these names). \
        You may propose MULTIPLE actions in a single turn by listing them all in the \
        \"actions\" array — e.g. read three files at once, or run several searches — \
        instead of waiting a full turn per call. Rules: search_codebase, read_file, and \
        list_directory may appear more than once per turn. run_command and \
        write_file_edits may each appear AT MOST ONCE per turn. final_answer, if used, \
        must be the ONLY action in its turn.\n\n\
        NOTE: \"action_input\" must be a JSON-ENCODED STRING containing the arguments \
        object, not a nested object. For example: \"action_input\": \"{\\\"path\\\": \\\"index.html\\\", \\\"start_line\\\": 1, \\\"end_line\\\": 100}\".\n\n\
        - search_codebase: arguments {\"query\": \"<search terms>\"} — full-text search the indexed codebase.\n\
        - read_file: arguments {\"path\": \"<file path>\", \"start_line\": <int>, \"end_line\": <int>} — read a bounded line range from a file. The field is \"path\", not \"file_path\".\n\
        - list_directory: arguments {\"path\": \"<relative dir path, use '.' for root>\"} — list files/subdirectories.\n\
        - run_command: arguments {\"cmd\": \"<shell command>\"} — run a shell command in the project root (e.g. to inspect the project or run tests).\n",
    );
    if allow_write {
        desc.push_str(
            "- write_file_edits: arguments {\"edits\": [{\"file_path\": \"...\", \"mode\": \"replace\"|\"append\"|\"create\", \"search\": \"...\", \"replace\": \"...\"}]} — propose edits to apply to disk. \
For \"replace\", \"search\" must be copied verbatim from a file you have actually read or retrieved. For \"append\" or \"create\", omit or empty \"search\" (for \"create\", \"replace\" holds the full contents of the new file, and \"file_path\" must not already exist). The user will be asked to confirm before anything is written. Note this action uses \"file_path\" inside each edit entry — that name does NOT apply to read_file.\n",
        );
    }
    desc.push_str(
        "- final_answer: arguments {\"answer\": \"<your final answer or summary to the user>\"} — call this ALONE, with nothing else in \"actions\", once you have nothing further to investigate or change.\n",
    );
    desc
}

const MAX_ACTIONS_PER_TURN: usize = 5;
const MAX_AGENT_STEPS: usize = 12;
const MAX_EDIT_BUILD_RETRIES: usize = 3;

#[derive(Debug, Clone)]
struct AgentStep {
    step_number: usize,
    thought: String,
    action: String,
    action_input: serde_json::Value,
    observation: String,
}

impl AgentStep {
    /// Renders a step exactly as it appeared in the old flat scratchpad, so the model
    /// sees a familiar transcript format regardless of how it's stored internally.
    fn render(&self) -> String {
        format!(
            "Thought: {}\nAction: {}\nAction Input: {}\nObservation:\n{}\n",
            self.thought, self.action, self.action_input, self.observation
        )
    }

    /// A compact one-line form used once a step ages out of the verbatim window,
    /// so it can still be swept into the periodic summary.
    fn render_compact(&self) -> String {
        let obs_preview: String = self.observation.chars().take(200).collect();
        let truncated = self.observation.chars().count() > 200;
        format!(
            "Step {}: action={} input={} -> {}{}",
            self.step_number,
            self.action,
            self.action_input,
            obs_preview.replace('\n', " "),
            if truncated { " [...]" } else { "" }
        )
    }
}

/// Bounded, structured replacement for the flat `scratchpad: String`.
///
/// Keeps the most recent `MAX_VERBATIM_STEPS` steps in full (thought + action +
/// full observation, exactly as the model needs them for near-term reasoning),
/// and periodically folds everything older than that into `summary` — a running,
/// human/LLM-authored recap — so the rendered transcript stays roughly bounded in
/// size no matter how many steps the agent takes.
struct AgentHistory {
    steps: Vec<AgentStep>,
    summary: String,
    next_step_number: usize,
    /// Number of steps folded into `summary` since the last summarization pass,
    /// used to decide when it's worth paying for another summarization call.
    steps_since_summary: usize,
}

/// How many of the most recent steps are always kept verbatim (full observations).
/// Chosen to comfortably cover the model's typical "what did I just learn" lookback
/// without keeping e.g. full 400-line file reads from 20 steps ago in every prompt.
const MAX_VERBATIM_STEPS: usize = 6;

/// Once more than this many steps have aged out of the verbatim window since the
/// last summarization pass, collapse them. Avoids re-summarizing on every single
/// step (which would add an extra model call per turn).
const SUMMARIZE_EVERY_N_STEPS: usize = 4;

impl AgentHistory {
    fn new() -> Self {
        Self {
            steps: Vec::new(),
            summary: String::new(),
            next_step_number: 1,
            steps_since_summary: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.steps.is_empty() && self.summary.is_empty()
    }

    /// Records a completed step. Call once per agent turn, after the tool has run
    /// and you have its observation text in hand.
    fn record_step(
        &mut self,
        thought: String,
        action: String,
        action_input: serde_json::Value,
        observation: String,
    ) {
        self.steps.push(AgentStep {
            step_number: self.next_step_number,
            thought,
            action,
            action_input,
            observation,
        });
        self.next_step_number += 1;
    }

    /// Appends a bare observation to the most recently recorded step (used for the
    /// "invalid JSON, try again" and similar meta-observations that don't map to a
    /// real tool call). Falls back to a synthetic step if history is empty.
    fn push_bare_observation(&mut self, observation: &str) {
        if let Some(last) = self.steps.last_mut() {
            last.observation.push('\n');
            last.observation.push_str(observation);
        } else {
            self.record_step(
                String::new(),
                "(none)".to_string(),
                serde_json::Value::Null,
                observation.to_string(),
            );
        }
    }

    /// Folds steps older than the verbatim window into `summary`. Uses a cheap
    /// heuristic compaction by default; if `use_llm` is true and enough steps have
    /// accumulated, asks the model itself to write a proper running summary instead
    /// (higher quality, costs one extra call — only triggered every
    /// `SUMMARIZE_EVERY_N_STEPS` steps, not every turn).
    async fn maybe_summarize(&mut self, use_gemini: bool, model: &str) {
        if self.steps.len() <= MAX_VERBATIM_STEPS {
            return;
        }

        let overflow = self.steps.len() - MAX_VERBATIM_STEPS;
        self.steps_since_summary += overflow.saturating_sub(
            self.steps
                .len()
                .saturating_sub(MAX_VERBATIM_STEPS + self.steps_since_summary),
        );

        if overflow < SUMMARIZE_EVERY_N_STEPS && !self.summary.is_empty() {
            // Not enough new overflow yet to be worth re-summarizing; just compact
            // the extra steps heuristically so the transcript doesn't keep growing
            // between summarization passes.
            self.compact_overflow_heuristically(overflow);
            return;
        }

        let to_fold: Vec<&AgentStep> = self.steps[..overflow].iter().collect();
        let folded_text: String = to_fold
            .iter()
            .map(|s| s.render_compact())
            .collect::<Vec<_>>()
            .join("\n");

        let new_summary = match self
            .try_llm_summarize(use_gemini, model, &folded_text)
            .await
        {
            Some(s) => s,
            None => {
                // LLM summarization unavailable/failed: fall back to heuristic
                // concatenation so we still bound growth without losing the run.
                format!(
                    "{}{}{}",
                    self.summary,
                    if self.summary.is_empty() { "" } else { "\n" },
                    folded_text
                )
            }
        };

        self.summary = new_summary;
        self.steps.drain(..overflow);
        self.steps_since_summary = 0;
    }

    /// Cheap fallback: just trims older observations down to their compact form
    /// in place, without an extra model call. Used between full summarization passes.
    fn compact_overflow_heuristically(&mut self, overflow: usize) {
        for step in self.steps.iter_mut().take(overflow) {
            if step.observation.chars().count() > 200 {
                let preview: String = step.observation.chars().take(200).collect();
                step.observation = format!("{} [...truncated, see summary...]", preview);
            }
        }
    }

    /// Asks the model to condense the given block of aged-out steps into a short
    /// running summary, merged with the existing summary. Returns `None` on any
    /// failure so the caller can fall back to heuristic compaction instead of
    /// derailing the agent loop over a summarization hiccup.
    async fn try_llm_summarize(
        &self,
        use_gemini: bool,
        model: &str,
        folded_text: &str,
    ) -> Option<String> {
        let prompt = format!(
            "Condense the following agent history into a short running summary (max ~150 words). \
            Preserve concrete facts the agent will still need: file paths touched, key findings from \
            search/read actions, build/test results, and any decisions made. Drop routine tool-call noise.\n\n\
            EXISTING SUMMARY:\n{}\n\n\
            NEW STEPS TO FOLD IN:\n{}\n\n\
            Respond with ONLY the updated summary text, no preamble.",
            if self.summary.is_empty() {
                "(none yet)"
            } else {
                &self.summary
            },
            folded_text
        );

        // Note: summarization is a free-form text task, not an agent "next action" turn,
        // so it deliberately does NOT pass agent_action_schema() here — that schema is
        // specific to the {thought, actions:[...]} contract used by the main loop.
        let result = if use_gemini {
            call_gemini_once(model, prompt, None).await
        } else {
            call_ollama_once(model, prompt, None).await
        };

        result
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// Renders the full transcript block to inject into the next prompt: the running
    /// summary (if any) followed by the verbatim recent steps.
    fn render(&self) -> String {
        if self.is_empty() {
            return "(nothing yet — this is your first step)".to_string();
        }

        let mut out = String::new();
        if !self.summary.is_empty() {
            out.push_str("Summary of earlier steps:\n");
            out.push_str(&self.summary);
            out.push_str("\n\n---\n\n");
        }
        for step in &self.steps {
            out.push_str(&step.render());
        }
        out
    }
}

/// The shared ReAct-style agent loop used by both `cix ask` and `cix edit`. At each step the
/// model is given the accumulated Thought/Action/Observation transcript and must respond with
/// exactly one JSON turn. The loop executes the requested tool(s), appends the result as an
/// Observation, and repeats until the model calls `final_answer` or the step budget runs out.
///
/// When `allow_write` is true (the `edit` case), a successful `write_file_edits` call is
/// automatically followed by the project's detected build/test command; a failure is fed back
/// to the model as an Observation so it can attempt a fix, up to `MAX_EDIT_BUILD_RETRIES` times.
async fn run_agent_loop(
    goal_description: &str,
    index: &Index,
    file_path_field: Field,
    content_field: Field,
    target_dir: &str,
    provider: &str,
    model: &str,
    allow_write: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let use_gemini = match provider.to_lowercase().as_str() {
        "gemini" => true,
        "ollama" | "local" => false,
        _ => model.to_lowercase().contains("gemini"),
    };

    let tools_desc = build_tools_description(allow_write);
    let system_preamble = format!(
        "You are an autonomous coding agent working inside a codebase.\n\n\
        GOAL:\n{}\n\n\
        Proceed step by step. At EVERY turn, respond with ONLY a single JSON object of the shape \
        {{\"thought\": \"<brief reasoning>\", \"actions\": [{{\"action\": \"<name>\", \"action_input\": \"<JSON-encoded string of arguments>\"}}, ...]}}.\n\
        Do not include any text outside the JSON object, and do not wrap it in markdown fences.\n\n\
        {}",
        goal_description, tools_desc
    );

    let mut history = AgentHistory::new();
    let mut build_retry_count = 0usize;
    let mut consecutive_parse_failures = 0usize;
    let mut consecutive_search_count = 0usize;

    // Tracks the last (action, action_input) pair executed, plus how many times in a row
    // it has repeated verbatim. Catches a stuck agent regardless of which action it's
    // stuck on (not just search_codebase), e.g. retrying a malformed read_file call.
    let mut last_call: Option<(String, serde_json::Value)> = None;
    let mut repeat_count = 0usize;
    const MAX_IDENTICAL_REPEATS: usize = 2;

    'turns: for step in 1..=MAX_AGENT_STEPS {
        history.maybe_summarize(use_gemini, model).await;
        let transcript = history.render();

        let prompt = format!(
            "{}\n\nTranscript so far:\n{}\n\nWhat are your next action(s)? Respond with ONLY the JSON object.",
            system_preamble, transcript
        );

        println!(
            "{}",
            format!(" [turn {}/{}] thinking...", step, MAX_AGENT_STEPS).dimmed()
        );

        let raw = if use_gemini {
            call_gemini_once(model, prompt, Some(agent_action_schema())).await?
        } else {
            call_ollama_once(model, prompt, Some(agent_action_schema())).await?
        };

        let cleaned = raw
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let parsed: Option<AgentTurn> = serde_json::from_str::<AgentTurnRaw>(cleaned)
            .ok()
            .or_else(|| {
                extract_json_object(&raw).and_then(|s| serde_json::from_str::<AgentTurnRaw>(s).ok())
            })
            .map(normalize_turn);

        let turn = match parsed {
            Some(t) if !t.actions.is_empty() => {
                consecutive_parse_failures = 0;
                t
            }
            _ => {
                consecutive_parse_failures += 1;
                if consecutive_parse_failures >= 3 {
                    eprintln!(
                        "{}",
                        " Agent repeatedly failed to produce a valid turn; stopping."
                            .red()
                            .bold()
                    );
                    return Ok(());
                }
                history.push_bare_observation(
                    "Observation: Your previous response was not valid JSON matching the required \
                    {\"thought\": ..., \"actions\": [...]} shape, or \"actions\" was empty. \
                    Respond with ONLY the JSON object, nothing else, and include at least one action. \
                    Remember: \"action_input\" must be a JSON-ENCODED STRING, not a nested object.",
                );
                continue;
            }
        };

        // --- Validate the turn before executing anything in it ---
        let final_count = turn
            .actions
            .iter()
            .filter(|a| a.action == "final_answer")
            .count();
        let write_count = turn
            .actions
            .iter()
            .filter(|a| a.action == "write_file_edits")
            .count();
        let run_count = turn
            .actions
            .iter()
            .filter(|a| a.action == "run_command")
            .count();

        if turn.actions.len() > MAX_ACTIONS_PER_TURN {
            history.push_bare_observation(&format!(
                "Observation: You proposed {} actions in one turn; the maximum is {}. \
                Split this into smaller turns.",
                turn.actions.len(),
                MAX_ACTIONS_PER_TURN
            ));
            continue;
        }
        if final_count > 0 && turn.actions.len() > 1 {
            history.push_bare_observation(
                "Observation: final_answer must be the ONLY action in its turn. \
                This turn was rejected — nothing in it was executed. Call final_answer \
                alone once you're ready to answer.",
            );
            continue;
        }
        if write_count > 1 {
            history.push_bare_observation(
                "Observation: write_file_edits may appear at most once per turn — combine \
                all your edits into that single call's \"edits\" array instead. \
                This turn was rejected — nothing in it was executed.",
            );
            continue;
        }
        if run_count > 1 {
            history.push_bare_observation(
                "Observation: run_command may appear at most once per turn. \
                This turn was rejected — nothing in it was executed.",
            );
            continue;
        }

        if let Some(t) = &turn.thought {
            if !t.trim().is_empty() {
                println!("  {} {}", "Thought:".dimmed(), t.dimmed());
            }
        }
        let thought = turn.thought.clone().unwrap_or_default();

        if turn.actions.len() > 1 {
            println!(
                "  {}",
                format!("(batched turn: {} actions)", turn.actions.len()).dimmed()
            );
        }

        // --- Execute every action in this turn, in order ---
        for proposed in &turn.actions {
            let action_name = proposed.action.clone();
            let action_input = proposed.action_input.clone();

            if DEBUG_ACTIONS {
                eprintln!(
                    "{}",
                    format!(
                        "  DEBUG: parsed action_input for '{}': {}",
                        action_name, action_input
                    )
                    .dimmed()
                );
            }

            // Repeat-call detection: if this exact (action, action_input) pair was also
            // the last thing executed, the agent is stuck (e.g. retrying a malformed
            // call without changing anything). A sharper, explicit nudge here matters
            // because the tool's own error text alone clearly wasn't enough to break
            // the loop — repeating the same error every turn doesn't teach the model
            // anything new about what to change.
            let is_repeat = last_call
                .as_ref()
                .map_or(false, |(a, i)| *a == action_name && *i == action_input);
            if is_repeat {
                repeat_count += 1;
            } else {
                repeat_count = 0;
            }
            last_call = Some((action_name.clone(), action_input.clone()));

            if repeat_count > MAX_IDENTICAL_REPEATS && action_name != "final_answer" {
                history.push_bare_observation(&format!(
                    "Observation: You have called {} with the EXACT SAME action_input ({}) \
                    {} times in a row and it keeps failing or returning the same result. \
                    Do not repeat this call unchanged again. Re-read the tool's argument \
                    names in the system prompt, change the input, try a different action, \
                    or call final_answer explaining you're stuck.",
                    action_name,
                    action_input,
                    repeat_count + 1
                ));
                repeat_count = 0;
                continue;
            }

            match action_name.as_str() {
                "final_answer" => {
                    let answer = match &action_input {
                        serde_json::Value::Object(_) => action_input
                            .get("answer")
                            .and_then(|v| v.as_str())
                            .map(str::to_string),
                        serde_json::Value::String(s) => Some(s.clone()),
                        _ => None,
                    }
                    .unwrap_or_else(|| "(no answer text provided)".to_string());
                
                    println!("\n{}", "Final answer:".bold().green());
                    println!("{}", answer);
                    return Ok(());
                }
                "search_codebase" => {
                    let query = action_input
                        .get("query")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    println!("  {} search_codebase(\"{}\")", "Action:".cyan(), query);
                    let obs = tool_search_codebase(query, index, file_path_field, content_field);
                    let no_results = obs.contains("No results found");

                    consecutive_search_count += 1;
                    history.record_step(thought.clone(), action_name, action_input, obs);

                    if consecutive_search_count >= 3 {
                        if no_results {
                            history.push_bare_observation(
                                "Observation: You have searched 3 times in a row with no results. \
                                Stop searching with these terms. Either try ONE very different, \
                                broader keyword, or call final_answer explaining that you could \
                                not locate this in the codebase.",
                            );
                        } else {
                            history.push_bare_observation(
                                "Observation: You have called search_codebase 3 times in a row. \
                                Pick the most relevant FILE path shown above and call read_file \
                                on it now — you can batch this with your remaining searches in \
                                the same turn.",
                            );
                        }
                        consecutive_search_count = 0;
                    }
                }
                "read_file" => {
                    consecutive_search_count = 0;
                    // Accept "path" (documented) but fall back to "file_path" (the name
                    // used by write_file_edits) since smaller models frequently conflate
                    // the two — this avoids most stuck-empty-path loops even if the
                    // model never fully self-corrects.
                    let path = action_input
                        .get("path")
                        .or_else(|| action_input.get("file_path"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let start = action_input
                        .get("start_line")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1) as usize;
                    let end = action_input
                        .get("end_line")
                        .and_then(|v| v.as_u64())
                        .unwrap_or((start + 100) as u64) as usize;
                    println!(
                        "  {} read_file(\"{}\", {}, {})",
                        "Action:".cyan(),
                        path,
                        start,
                        end
                    );
                    let obs = tool_read_file(target_dir, path, start, end, &action_input);
                    history.record_step(thought.clone(), action_name, action_input, obs);
                }
                "list_directory" => {
                    consecutive_search_count = 0;
                    let path = action_input
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or(".");
                    println!("  {} list_directory(\"{}\")", "Action:".cyan(), path);
                    let obs = tool_list_directory(target_dir, path);
                    history.record_step(thought.clone(), action_name, action_input, obs);
                }
                "run_command" => {
                    consecutive_search_count = 0;
                    let cmd = action_input
                        .get("cmd")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let obs = if cmd.trim().is_empty() {
                        "Error: no command provided.".to_string()
                    } else {
                        let result = tool_run_command(target_dir, cmd).await;
                        format!(
                            "({}):\n{}",
                            if result.success { "success" } else { "failed" },
                            result.text
                        )
                    };
                    history.record_step(thought.clone(), action_name, action_input, obs);
                }
                "write_file_edits" if allow_write => {
                    consecutive_search_count = 0;
                    let edits_val = action_input
                        .get("edits")
                        .cloned()
                        .unwrap_or(serde_json::Value::Array(vec![]));
                    let edits = parse_edit_blocks_from_value(&edits_val);
                    let (mut obs, applied) = propose_and_apply_edits(edits, target_dir);
                    println!("  {} {}", "Observation:".dimmed(), obs.dimmed());

                    if applied {
                        if let Some(build_cmd) = detect_build_command(target_dir) {
                            if build_retry_count < MAX_EDIT_BUILD_RETRIES {
                                println!(
                                    "{}",
                                    format!(" Verifying with build/test check: {}", build_cmd)
                                        .cyan()
                                );
                                let result = tool_run_command(target_dir, &build_cmd).await;
                                build_retry_count += 1;

                                if result.success {
                                    obs.push_str(&format!(
                                        "\nBuild check '{}', attempt {}/{}: PASSED.\n{}",
                                        build_cmd,
                                        build_retry_count,
                                        MAX_EDIT_BUILD_RETRIES,
                                        result.text
                                    ));
                                    println!("{}", " Build/test check passed.".green());
                                } else {
                                    obs.push_str(&format!(
                                        "\nBuild check '{}', attempt {}/{}: FAILED.\n{}\n\
                                        Please analyze this error and propose a fix using write_file_edits, \
                                        or call final_answer explaining the issue if you cannot resolve it.",
                                        build_cmd, build_retry_count, MAX_EDIT_BUILD_RETRIES, result.text
                                    ));
                                    println!(
                                        "{}",
                                        " Build/test check failed; feeding error back to the agent.".yellow()
                                    );
                                }
                            } else {
                                obs.push_str(
                                    "\nMaximum build/test retry attempts reached; not running the build check again.",
                                );
                            }
                        }
                    }

                    history.record_step(thought.clone(), action_name, action_input, obs);
                }
                other => {
                    consecutive_search_count = 0;
                    let obs = format!(
                        "Unknown action '{}'. Choose one of the actions listed in the system prompt.",
                        other
                    );
                    history.record_step(thought.clone(), action_name, action_input, obs);
                }
            }
        }

        let _ = step; // step index used only for the progress message above
        continue 'turns;
    }

    println!(
        "{}",
        " Reached the maximum number of agent turns without a final answer.".yellow()
    );
    Ok(())
}

/// Sets up the index and runs the read-only agent loop to answer a question about the codebase.
async fn run_ask_agent_pipeline(
    question: &str,
    model: &str,
    provider: &str,
    target_dir: &str,
    app_cache_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "Starting agent to answer your question...".cyan());
    let (index, file_path_field, content_field) = ensure_index(target_dir, false, app_cache_dir)?;

    let goal = format!(
        "Answer the user's question about this codebase as accurately as possible. \
        Use search_codebase, read_file, list_directory, and run_command as needed to gather real \
        evidence before answering — do not guess. If the answer isn't in the codebase, say so.\n\n\
        USER QUESTION:\n{}",
        question
    );

    run_agent_loop(
        &goal,
        &index,
        file_path_field,
        content_field,
        target_dir,
        provider,
        model,
        false,
    )
    .await
}

/// Sets up the index and runs the write-enabled agent loop to carry out a code edit,
/// automatically verifying the result with a build/test check when possible.
async fn run_edit_agent_pipeline(
    instruction: &str,
    model: &str,
    provider: &str,
    target_dir: &str,
    app_cache_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        "Starting agent to perform the requested edit...".cyan()
    );
    let (index, file_path_field, content_field) = ensure_index(target_dir, false, app_cache_dir)?;

    let goal = format!(
        "Modify the codebase to satisfy the following instruction. Investigate with \
        search_codebase / read_file / list_directory first so your edits are grounded in the \
        real file contents. If the directory is empty or the relevant files don't exist yet, \
        don't keep re-listing it — go straight to write_file_edits with mode \"create\" to \
        author the needed files. After edits are applied you will automatically be shown the \
        result of the project's build/test check — if it fails, analyze the error and propose \
        a fix. Call final_answer once the change is complete (or once you've made a best effort \
        and should report status/blockers).\n\n\
        INSTRUCTION:\n{}",
        instruction
    );

    run_agent_loop(
        &goal,
        &index,
        file_path_field,
        content_field,
        target_dir,
        provider,
        model,
        true,
    )
    .await
}

async fn run_one_shot_ask_pipeline(
    question: &str,
    model: &str,
    _provider: &str,
    target_dir: &str,
    app_cache_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (index, file_path_field, content_field) = ensure_index(target_dir, false, app_cache_dir)?;

    let keywords = extract_keywords(question);
    let context_snippets = tool_search_codebase(&keywords, &index, file_path_field, content_field);

    let prompt = format!(
        "You are a helpful coding assistant. Answer the user's question based on the provided codebase context.\n\n\
        CODEBASE CONTEXT:\n{}\n\n\
        USER QUESTION:\n{}\n",
        context_snippets, question
    );

    println!("{}", "Sending one-shot query to LLM...".cyan());

    let response = call_gemini_once(model, prompt, None).await?;

    println!("\n{}", "Answer:".bold().green());
    println!("{}", response);

    Ok(())
}

async fn run_one_shot_edit_pipeline(
    instruction: &str,
    model: &str,
    provider: &str,
    target_dir: &str,
    app_cache_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "Starting 1-Shot Edit Pipeline...".cyan());

    let (index, file_path_field, content_field) = ensure_index(target_dir, false, app_cache_dir)?;
    let search_terms = extract_keywords(instruction);
    let codebase_context =
        tool_search_codebase(&search_terms, &index, file_path_field, content_field);

    let prompt = format!(
        "You are an automated code editing tool. Your task is to modify the codebase to satisfy the following instruction.\n\n\
        INSTRUCTION:\n{}\n\n\
        RELEVANT CODEBASE CONTEXT:\n{}\n\n\
        OUTPUT FORMAT:\n\
        Output your edits using EXACT SEARCH/REPLACE blocks formatted like this:\n\
        FILE: path/to/file.ext\n\
        <<<<<<< SEARCH\n\
        verbatim text to replace\n\
        =======\n\
        new replacement text\n\
        >>>>>>> REPLACE\n\n\
        To CREATE a brand new file, leave the SEARCH section empty and put the full file contents in REPLACE:\n\
        FILE: path/to/new_file.ext\n\
        <<<<<<< SEARCH\n\
        =======\n\
        full contents of the new file\n\
        >>>>>>> REPLACE\n\n\
        Important:\n\
        - The SEARCH block must match existing code EXACTLY line-by-line (or be empty, for new files).\n\
        - Output ONLY the edit blocks. No conversational text.",
        instruction, codebase_context
    );

    let use_gemini = provider.to_lowercase() == "gemini" || model.contains("gemini");
    let response = if use_gemini {
        call_gemini_once(model, prompt, None).await?
    } else {
        call_ollama_once(model, prompt, None).await?
    };

    let edit_blocks = parse_edit_blocks(&response);
    if edit_blocks.is_empty() {
        println!(
            "{}",
            "No valid SEARCH/REPLACE blocks were produced by the model.".yellow()
        );
        println!("\nRaw response:\n{}", response);
        return Ok(());
    }

    let (obs, applied) = propose_and_apply_edits(edit_blocks, target_dir);
    println!("\n{}", obs.bold());

    if applied {
        if let Some(build_cmd) = detect_build_command(target_dir) {
            println!(
                "{}",
                format!(" Running verification build: {}", build_cmd).cyan()
            );
            let result = tool_run_command(target_dir, &build_cmd).await;
            if result.success {
                println!("{}", " Build passed successfully!".green().bold());
            } else {
                println!(
                    "{}",
                    " Build failed after applying 1-shot edits:".red().bold()
                );
                println!("{}", result.text);
            }
        }
    }

    Ok(())
}

fn resolve_new_file_path(proposed_path: &str, target_dir: &str) -> Option<std::path::PathBuf> {
    let clean_proposed = proposed_path.trim_start_matches(r"\\?\").replace('\\', "/");
    let path = std::path::Path::new(&clean_proposed);

    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::path::Path::new(target_dir).join(&clean_proposed)
    };

    if candidate.exists() {
        None // already exists — use "replace" or "append" instead of "create"
    } else {
        Some(candidate)
    }
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
    let query = parse_query(&query_parser, query_arg);

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
    Create,
}

/// Parses marker-format SEARCH/REPLACE blocks from a text response. Kept for callers (and
/// tests) that still want to parse the plain marker format directly; the live agent loop uses
/// `parse_edit_blocks_from_value` on structured JSON instead.
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
            if !current_file.is_empty() && !replace_lines.is_empty() {
                let mode = if search_lines.is_empty() {
                    EditMode::Create
                } else {
                    EditMode::Replace
                };
                blocks.push(EditBlock {
                    file_path: current_file.clone(),
                    mode,
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
        let file_path = match block.mode {
            EditMode::Create => match resolve_new_file_path(&block.file_path, target_dir) {
                Some(p) => p,
                None => {
                    eprintln!(
                        "{}",
                        format!(
                            " Warning: refusing to create {} — already exists",
                            block.file_path
                        )
                        .yellow()
                    );
                    continue;
                }
            },
            EditMode::Replace | EditMode::Append => {
                match resolve_file_path(&block.file_path, target_dir) {
                    Some(p) => p,
                    None => {
                        eprintln!(
                            "{}",
                            format!(" Warning: File not found: {}", block.file_path).yellow()
                        );
                        continue;
                    }
                }
            }
        };

        match block.mode {
            EditMode::Create => {
                if let Some(parent) = file_path.parent() {
                    if let Err(e) = fs::create_dir_all(parent) {
                        eprintln!(
                            "{}",
                            format!(
                                " Warning: could not create directory {}: {}",
                                parent.display(),
                                e
                            )
                            .yellow()
                        );
                        continue;
                    }
                }
                let content = block.replace.replace("\r\n", "\n");
                fs::write(&file_path, content)?;
                applied_count += 1;
            }
            EditMode::Replace => {
                let content = match fs::read_to_string(&file_path) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!(
                            "{}",
                            format!(" Warning: could not read {}: {}", file_path.display(), e)
                                .yellow()
                        );
                        continue;
                    }
                };
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
                let content = match fs::read_to_string(&file_path) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!(
                            "{}",
                            format!(" Warning: could not read {}: {}", file_path.display(), e)
                                .yellow()
                        );
                        continue;
                    }
                };
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
/// Constructs a Tantivy query from user input using plain keyword extraction.
fn parse_query(query_parser: &QueryParser, input_text: &str) -> Box<dyn tantivy::query::Query> {
    let keywords = extract_keywords(input_text);
    if keywords.trim().is_empty() {
        return query_parser
            .parse_query("*")
            .unwrap_or_else(|_| query_parser.parse_query("").unwrap());
    }

    if let Ok(q) = query_parser.parse_query(&keywords) {
        return q;
    }

    query_parser.parse_query("*").unwrap()
}
/// Checks if a line matches a keyword either as a substring or via fuzzy matching.
fn line_matches_keyword(line_lower: &str, kw: &str) -> bool {
    line_lower.contains(kw)
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
    fn test_parse_edit_blocks_from_value() {
        let v = serde_json::json!({
            "edits": [
                {"file_path": "src/a.rs", "mode": "replace", "search": "foo", "replace": "bar"},
                {"file_path": "src/b.rs", "mode": "append", "replace": "// new line"},
                {"file_path": "src/c.rs", "mode": "replace", "replace": "no search, dropped"}
            ]
        });
        let blocks = parse_edit_blocks_from_value(&v["edits"]);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].mode, EditMode::Replace);
        assert_eq!(blocks[1].mode, EditMode::Append);
    }

    #[test]
    fn test_extract_json_object() {
        let text = "Sure, here you go:\n```json\n{\"action\": \"final_answer\", \"action_input\": {}}\n```\nlet me know if that helps";
        let extracted = extract_json_object(text).unwrap();
        let parsed: ProposedActionRaw = serde_json::from_str(extracted).unwrap();
        assert_eq!(parsed.action, "final_answer");
    }

    #[test]
    fn test_normalize_turn_string_action_input() {
        // Simulates Gemini's constrained shape: action_input as a JSON-encoded string.
        let raw = AgentTurnRaw {
            thought: Some("reading a file".to_string()),
            actions: vec![ProposedActionRaw {
                action: "read_file".to_string(),
                action_input: serde_json::Value::String(
                    "{\"path\": \"index.html\", \"start_line\": 1, \"end_line\": 50}".to_string(),
                ),
            }],
        };
        let turn = normalize_turn(raw);
        assert_eq!(turn.actions.len(), 1);
        assert_eq!(
            turn.actions[0]
                .action_input
                .get("path")
                .and_then(|v| v.as_str()),
            Some("index.html")
        );
    }

    #[test]
    fn test_normalize_turn_object_action_input() {
        // Simulates Ollama's looser shape: action_input already a real object.
        let raw = AgentTurnRaw {
            thought: None,
            actions: vec![ProposedActionRaw {
                action: "list_directory".to_string(),
                action_input: serde_json::json!({"path": "."}),
            }],
        };
        let turn = normalize_turn(raw);
        assert_eq!(
            turn.actions[0]
                .action_input
                .get("path")
                .and_then(|v| v.as_str()),
            Some(".")
        );
    }

    #[test]
    fn test_agent_action_schema() {
        let schema = agent_action_schema();
        assert!(schema.is_object());
        let obj = schema.as_object().unwrap();
        assert!(obj.contains_key("properties"));
    }

    #[test]
    fn test_detect_build_command() {
        let cmd = detect_build_command(".");
        assert_eq!(cmd, Some("cargo check --message-format short".to_string()));

        let none_cmd = detect_build_command("nonexistent_dir_abc_123");
        assert_eq!(none_cmd, None);
    }

    #[test]
    fn test_line_matches_keyword() {
        assert!(line_matches_keyword("fn process_items()", "process"));
        assert!(line_matches_keyword("fn process_items()", "proces"));
        assert!(!line_matches_keyword(
            "fn process_items()",
            "completelyunrelated"
        ));
    }

    #[test]
    fn test_tool_list_directory_and_read_file() {
        let listing = tool_list_directory(".", ".");
        assert!(listing.contains("Cargo.toml"));
        assert!(listing.contains("src"));

        let read_res = tool_read_file(".", "Cargo.toml", 1, 5, &serde_json::Value::Null);
        assert!(read_res.contains("[package]") || read_res.contains("name"));

        let err_read = tool_read_file(
            ".",
            "nonexistent_file_xyz.rs",
            1,
            10,
            &serde_json::Value::Null,
        );
        assert!(err_read.contains("Error"));
    }
}

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

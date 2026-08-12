//! Semantic code chunking for `cix`.

use tree_sitter::{Node, Parser};

/// Soft cap on characters per chunk. Chosen as a rough proxy for staying
/// well under typical local-model context budgets when several chunks are
/// concatenated into a RAG prompt. Tune per model if needed.
const MAX_CHUNK_CHARS: usize = 3_500;

/// Below this size we don't bother trying to split further even if a natural
/// boundary exists nearby — avoids fragmenting small items.
const MIN_SPLIT_CHARS: usize = 800;

/// Line overlap applied between consecutive fallback-mode chunks so content
/// straddling a cut boundary still appears whole in at least one chunk.
const OVERLAP_LINES: usize = 2;

#[derive(Debug, Clone, PartialEq)]
pub struct CodeChunk {
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// Entry point. Dispatches to a language-aware chunker based on file
/// extension, falling back to the brace-aware generic chunker for anything
/// without a tree-sitter grammar wired up (or if parsing fails).
pub fn chunk_code(content: &str, file_ext: &str) -> Vec<CodeChunk> {
    if content.trim().is_empty() {
        return vec![CodeChunk {
            content: String::new(),
            start_line: 1,
            end_line: 1,
        }];
    }

    let chunks = match file_ext.to_lowercase().as_str() {
        "rs" => chunk_rust(content),
        "cs" => chunk_c_sharp(content),
        // C/C++ extensions (source + headers)
        "c" | "cpp" | "cc" | "cxx" | "c++" | "h" | "hpp" | "hh" | "h++" | "hxx" => {
            chunk_cpp(content)
        }
        _ => None,
    };

    chunks.unwrap_or_else(|| chunk_generic_fallback(content))
}


fn is_item_node(kind: &str) -> bool {
    matches!(
        kind,
        "function_item"
            | "struct_item"
            | "enum_item"
            | "impl_item"
            | "trait_item"
            | "mod_item"
            | "macro_definition"
            | "const_item"
            | "static_item"
            | "type_item"
            | "union_item"
    )
}

fn is_leading_trivia(kind: &str) -> bool {
    matches!(kind, "line_comment" | "block_comment" | "attribute_item")
}

fn is_cs_item_node(kind: &str) -> bool {
    matches!(
        kind,
        "class_declaration"
            | "struct_declaration"
            | "interface_declaration"
            | "enum_declaration"
            | "namespace_declaration"
            | "file_scoped_namespace_declaration"
            | "method_declaration"
            | "property_declaration"
            | "field_declaration"
            | "constructor_declaration"
            | "destructor_declaration"
            | "delegate_declaration"
            | "event_declaration"
            | "event_field_declaration"
            | "record_declaration"
            | "using_directive"
    )
}

fn is_cs_leading_trivia(kind: &str) -> bool {
    matches!(
        kind,
        "comment" | "line_comment" | "block_comment" | "attribute_list"
    )
}

pub fn chunk_c_sharp(content: &str) -> Option<Vec<CodeChunk>> {
    let mut parser = Parser::new();

    let language = tree_sitter_c_sharp::language();
    parser.set_language(&language).ok()?;

    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    if root.has_error() {
        return None;
    }

    let mut chunks = Vec::new();
    let mut cursor = root.walk();
    let children: Vec<Node<'_>> = root.children(&mut cursor).collect();

    let mut i = 0usize;
    let mut pending_trivia_start: Option<usize> = None;

    while i < children.len() {
        let node = children[i];
        let kind = node.kind();

        if is_cs_leading_trivia(kind) {
            if pending_trivia_start.is_none() {
                pending_trivia_start = Some(node.start_byte());
            }
            i += 1;
            continue;
        }

        if is_cs_item_node(kind) {
            let start_byte = pending_trivia_start
                .take()
                .unwrap_or_else(|| node.start_byte());
            let end_byte = node.end_byte();
            push_cs_item_chunk(content, node, start_byte, end_byte, &mut chunks);
        } else {
            pending_trivia_start = None;
            let start_byte = node.start_byte();
            let end_byte = node.end_byte();
            emit_chunk_from_bytes(content, start_byte, end_byte, &mut chunks);
        }

        i += 1;
    }

    if let Some(start_byte) = pending_trivia_start {
        emit_chunk_from_bytes(content, start_byte, content.len(), &mut chunks);
    }

    if chunks.is_empty() {
        None
    } else {
        Some(chunks)
    }
}

fn push_cs_item_chunk(
    source: &str,
    node: Node<'_>,
    start_byte: usize,
    end_byte: usize,
    chunks: &mut Vec<CodeChunk>,
) {
    let span_len = end_byte.saturating_sub(start_byte);

    if span_len <= MAX_CHUNK_CHARS {
        emit_chunk_from_bytes(source, start_byte, end_byte, chunks);
        return;
    }

    match node.kind() {
        "class_declaration"
        | "struct_declaration"
        | "interface_declaration"
        | "namespace_declaration"
        | "record_declaration" => {
            split_cs_container_by_inner_items(source, node, chunks);
        }
        _ => {
            let text = &source[start_byte..end_byte];
            let base_line = byte_to_line(source, start_byte);
            chunks.extend(soft_split_by_braces(text, base_line));
        }
    }
}

fn split_cs_container_by_inner_items(source: &str, node: Node<'_>, chunks: &mut Vec<CodeChunk>) {
    let mut cursor = node.walk();
    let decl_list = node
        .children(&mut cursor)
        .find(|c| c.kind() == "declaration_list");

    let header_end = decl_list
        .map(|b| b.start_byte())
        .unwrap_or(node.start_byte());
    let header = source[node.start_byte()..header_end].trim_end();

    let Some(body) = decl_list else {
        emit_chunk_from_bytes(source, node.start_byte(), node.end_byte(), chunks);
        return;
    };

    let mut body_cursor = body.walk();
    let members: Vec<Node<'_>> = body.children(&mut body_cursor).collect();
    let mut pending_trivia_start: Option<usize> = None;

    for member in members {
        let kind = member.kind();
        if is_cs_leading_trivia(kind) {
            if pending_trivia_start.is_none() {
                pending_trivia_start = Some(member.start_byte());
            }
            continue;
        }
        if kind == "{" || kind == "}" {
            continue;
        }

        let start_byte = pending_trivia_start
            .take()
            .unwrap_or_else(|| member.start_byte());
        let end_byte = member.end_byte();
        let member_text = &source[start_byte..end_byte];

        let combined = format!("{}\n    // ...\n{}", header, member_text);
        let base_line = byte_to_line(source, start_byte);

        if combined.len() > MAX_CHUNK_CHARS {
            chunks.extend(soft_split_by_braces(&combined, base_line));
        } else {
            chunks.push(CodeChunk {
                content: combined.to_string(),
                start_line: base_line,
                end_line: base_line + member_text.lines().count().saturating_sub(1),
            });
        }
    }
}

fn is_cpp_item_node(kind: &str) -> bool {
    matches!(
        kind,
        "function_definition"
            | "class_specifier"
            | "struct_specifier"
            | "enum_specifier"
            | "union_specifier"
            | "namespace_definition"
            | "template_declaration"
            | "declaration"
            | "type_definition"
            | "using_declaration"
            | "alias_declaration"
            | "linkage_specification"
            | "concept_definition"
            | "preproc_include"
            | "preproc_def"
            | "preproc_function_def"
            | "preproc_ifdef"
            | "preproc_if"
    )
}

/// Leading comments and C++ attributes that belong with the following declaration.
fn is_cpp_leading_trivia(kind: &str) -> bool {
    matches!(
        kind,
        "comment" | "attribute_declaration" | "attribute_specifier"
    )
}

pub fn chunk_cpp(content: &str) -> Option<Vec<CodeChunk>> {
    let mut parser = Parser::new();
    let language = tree_sitter_cpp::language().into();
    parser.set_language(&language).ok()?;

    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    if root.has_error() {
        // Fall back to generic chunker on mid-edit parse failure
        return None;
    }

    let mut chunks = Vec::new();
    let mut cursor = root.walk();
    let children: Vec<Node<'_>> = root.children(&mut cursor).collect();

    let mut i = 0usize;
    let mut pending_trivia_start: Option<usize> = None;

    while i < children.len() {
        let node = children[i];
        let kind = node.kind();

        if is_cpp_leading_trivia(kind) {
            if pending_trivia_start.is_none() {
                pending_trivia_start = Some(node.start_byte());
            }
            i += 1;
            continue;
        }

        if is_cpp_item_node(kind) {
            let start_byte = pending_trivia_start
                .take()
                .unwrap_or_else(|| node.start_byte());
            let end_byte = node.end_byte();
            push_cpp_item_chunk(content, node, start_byte, end_byte, &mut chunks);
        } else {
            pending_trivia_start = None;
            let start_byte = node.start_byte();
            let end_byte = node.end_byte();
            emit_chunk_from_bytes(content, start_byte, end_byte, &mut chunks);
        }

        i += 1;
    }

    if let Some(start_byte) = pending_trivia_start {
        emit_chunk_from_bytes(content, start_byte, content.len(), &mut chunks);
    }

    if chunks.is_empty() {
        None
    } else {
        Some(chunks)
    }
}

/// Emits small C/C++ items or splits oversized classes, structs, or namespaces.
fn push_cpp_item_chunk(
    source: &str,
    node: Node<'_>,
    start_byte: usize,
    end_byte: usize,
    chunks: &mut Vec<CodeChunk>,
) {
    let span_len = end_byte.saturating_sub(start_byte);

    if span_len <= MAX_CHUNK_CHARS {
        emit_chunk_from_bytes(source, start_byte, end_byte, chunks);
        return;
    }

    match node.kind() {
        "class_specifier"
        | "struct_specifier"
        | "namespace_definition"
        | "linkage_specification"
        | "template_declaration" => {
            split_cpp_container_by_inner_items(source, node, chunks);
        }
        _ => {
            let text = &source[start_byte..end_byte];
            let base_line = byte_to_line(source, start_byte);
            chunks.extend(soft_split_by_braces(text, base_line));
        }
    }
}

/// Splits oversized C++ classes/namespaces per-method while preserving the class signature.
fn split_cpp_container_by_inner_items(
    source: &str,
    node: Node<'_>,
    chunks: &mut Vec<CodeChunk>,
) {
    let mut cursor = node.walk();
    let body_node = node.children(&mut cursor).find(|c| {
        matches!(
            c.kind(),
            "field_declaration_list" | "declaration_list" | "compound_statement"
        )
    });

    let header_end = body_node.map(|b| b.start_byte()).unwrap_or(node.start_byte());
    let header = source[node.start_byte()..header_end].trim_end();

    let Some(body) = body_node else {
        emit_chunk_from_bytes(source, node.start_byte(), node.end_byte(), chunks);
        return;
    };

    let mut body_cursor = body.walk();
    let members: Vec<Node<'_>> = body.children(&mut body_cursor).collect();
    let mut pending_trivia_start: Option<usize> = None;

    for member in members {
        let kind = member.kind();
        if is_cpp_leading_trivia(kind) {
            if pending_trivia_start.is_none() {
                pending_trivia_start = Some(member.start_byte());
            }
            continue;
        }
        if kind == "{" || kind == "}" || kind == ";" || kind == "access_specifier" {
            continue;
        }

        let start_byte = pending_trivia_start
            .take()
            .unwrap_or_else(|| member.start_byte());
        let end_byte = member.end_byte();
        let member_text = &source[start_byte..end_byte];

        let combined = format!("{}\n    // ...\n{}", header, member_text);
        let base_line = byte_to_line(source, start_byte);

        if combined.len() > MAX_CHUNK_CHARS {
            chunks.extend(soft_split_by_braces(&combined, base_line));
        } else {
            chunks.push(CodeChunk {
                content: combined.to_string(),
                start_line: base_line,
                end_line: base_line + member_text.lines().count().saturating_sub(1),
            });
        }
    }
}

fn chunk_rust(content: &str) -> Option<Vec<CodeChunk>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .ok()?;
    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    if root.has_error() {
        return None;
    }

    let mut chunks = Vec::new();
    let mut cursor = root.walk();
    let children: Vec<Node> = root.children(&mut cursor).collect();

    let mut i = 0usize;
    let mut pending_trivia_start: Option<usize> = None; // byte offset

    while i < children.len() {
        let node = children[i];
        let kind = node.kind();

        if is_leading_trivia(kind) {
            if pending_trivia_start.is_none() {
                pending_trivia_start = Some(node.start_byte());
            }
            i += 1;
            continue;
        }

        if is_item_node(kind) {
            let start_byte = pending_trivia_start
                .take()
                .unwrap_or_else(|| node.start_byte());
            let end_byte = node.end_byte();
            push_item_chunk(content, node, start_byte, end_byte, &mut chunks);
        } else {
            pending_trivia_start = None;
            let start_byte = node.start_byte();
            let end_byte = node.end_byte();
            emit_chunk_from_bytes(content, start_byte, end_byte, &mut chunks);
        }

        i += 1;
    }

    if let Some(start_byte) = pending_trivia_start {
        emit_chunk_from_bytes(content, start_byte, content.len(), &mut chunks);
    }

    if chunks.is_empty() {
        None
    } else {
        Some(chunks)
    }
}

fn push_item_chunk(
    source: &str,
    node: Node,
    start_byte: usize,
    end_byte: usize,
    chunks: &mut Vec<CodeChunk>,
) {
    let span_len = end_byte.saturating_sub(start_byte);

    if span_len <= MAX_CHUNK_CHARS {
        emit_chunk_from_bytes(source, start_byte, end_byte, chunks);
        return;
    }

    match node.kind() {
        "impl_item" | "trait_item" | "mod_item" => {
            split_container_by_inner_items(source, node, chunks);
        }
        _ => {
            let text = &source[start_byte..end_byte];
            let base_line = byte_to_line(source, start_byte);
            chunks.extend(soft_split_by_braces(text, base_line));
        }
    }
}

fn split_container_by_inner_items(source: &str, node: Node, chunks: &mut Vec<CodeChunk>) {
    let header_end = find_body_start(node, source).unwrap_or(node.start_byte());
    let header = source[node.start_byte()..header_end].trim_end();

    let Some(body) = node.child_by_field_name("body") else {
        emit_chunk_from_bytes(source, node.start_byte(), node.end_byte(), chunks);
        return;
    };

    let mut cursor = body.walk();
    let members: Vec<Node> = body.children(&mut cursor).collect();

    let mut pending_trivia_start: Option<usize> = None;

    for member in members {
        let kind = member.kind();
        if is_leading_trivia(kind) {
            if pending_trivia_start.is_none() {
                pending_trivia_start = Some(member.start_byte());
            }
            continue;
        }
        if kind == "{" || kind == "}" {
            continue;
        }

        let start_byte = pending_trivia_start
            .take()
            .unwrap_or_else(|| member.start_byte());
        let end_byte = member.end_byte();
        let member_text = &source[start_byte..end_byte];

        let combined = format!("{}\n    // ...\n{}", header, member_text);
        let base_line = byte_to_line(source, start_byte);

        if combined.len() > MAX_CHUNK_CHARS {
            chunks.extend(soft_split_by_braces(&combined, base_line));
        } else {
            chunks.push(CodeChunk {
                content: combined.to_string(),
                start_line: base_line,
                end_line: base_line + member_text.lines().count().saturating_sub(1),
            });
        }
    }
}

fn find_body_start(node: Node, _source: &str) -> Option<usize> {
    node.child_by_field_name("body").map(|b| b.start_byte())
}

fn emit_chunk_from_bytes(
    source: &str,
    start_byte: usize,
    end_byte: usize,
    chunks: &mut Vec<CodeChunk>,
) {
    if start_byte >= end_byte {
        return;
    }
    let text = source[start_byte..end_byte].trim_end_matches('\n');
    if text.trim().is_empty() {
        return;
    }
    let start_line = byte_to_line(source, start_byte);
    let end_line = start_line + text.lines().count().saturating_sub(1);
    chunks.push(CodeChunk {
        content: text.to_string(),
        start_line,
        end_line,
    });
}

fn byte_to_line(source: &str, byte_offset: usize) -> usize {
    source[..byte_offset.min(source.len())]
        .matches('\n')
        .count()
        + 1
}

fn soft_split_by_braces(text: &str, base_line: usize) -> Vec<CodeChunk> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut start_idx = 0usize;
    let mut depth: i64 = 0;

    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let current_len: usize = lines[start_idx..=i].iter().map(|l| l.len() + 1).sum();

        if current_len > MAX_CHUNK_CHARS && i > start_idx {
            let split_at = find_soft_break(&lines, start_idx, i, depth);
            let end = split_at.max(start_idx);
            push_offset_chunk(&lines, start_idx, end, base_line, &mut out);
            start_idx = end.saturating_sub(OVERLAP_LINES).max(start_idx) + 1;
        }

        depth += line.matches('{').count() as i64;
        depth -= line.matches('}').count() as i64;
        i += 1;
    }

    if start_idx < lines.len() {
        push_offset_chunk(&lines, start_idx, lines.len() - 1, base_line, &mut out);
    }

    out
}

fn push_offset_chunk(
    lines: &[&str],
    start_idx: usize,
    end_idx: usize,
    base_line: usize,
    out: &mut Vec<CodeChunk>,
) {
    if start_idx > end_idx || start_idx >= lines.len() {
        return;
    }
    let end_idx = end_idx.min(lines.len() - 1);
    let text = lines[start_idx..=end_idx].join("\n");
    if text.trim().is_empty() {
        return;
    }
    out.push(CodeChunk {
        content: text,
        start_line: base_line + start_idx,
        end_line: base_line + end_idx,
    });
}

fn chunk_generic_fallback(content: &str) -> Vec<CodeChunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return vec![CodeChunk {
            content: String::new(),
            start_line: 1,
            end_line: 1,
        }];
    }

    if content.len() <= MAX_CHUNK_CHARS {
        return vec![CodeChunk {
            content: content.to_string(),
            start_line: 1,
            end_line: lines.len(),
        }];
    }

    let boundary_keywords = [
        "pub ",
        "fn ",
        "async fn ",
        "struct ",
        "enum ",
        "impl ",
        "class ",
        "def ",
        "function ",
        "interface ",
        "trait ",
        "type ",
        "namespace ",
        "mod ",
        "match ",
        "public ",
        "private ",
        "protected ",
        "static ",
    ];
    let comment_prefixes = ["//", "#", "/*", "*", "'''", "\"\"\""];

    let mut chunks = Vec::new();
    let mut chunk_start_idx = 0usize;
    let mut pending_trivia_start: Option<usize> = None;
    let mut depth: i64 = 0;
    let mut chunk_min_depth: i64 = 0;

    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        let indent_depth = depth;

        let is_comment_like = comment_prefixes.iter().any(|p| trimmed.starts_with(p));
        if is_comment_like && pending_trivia_start.is_none() && line != lines[chunk_start_idx] {
            pending_trivia_start = Some(i);
        } else if !is_comment_like && !trimmed.is_empty() {
            pending_trivia_start = None;
        }

        let is_boundary_candidate = boundary_keywords.iter().any(|kw| trimmed.starts_with(kw))
            && indent_depth <= chunk_min_depth
            && i > chunk_start_idx;

        let current_len: usize = lines[chunk_start_idx..i].iter().map(|l| l.len() + 1).sum();

        if is_boundary_candidate && current_len >= MIN_SPLIT_CHARS {
            let split_at = pending_trivia_start.unwrap_or(i);
            if split_at > chunk_start_idx {
                push_fallback_chunk(&lines, chunk_start_idx, split_at - 1, &mut chunks);
                chunk_start_idx = split_at.saturating_sub(OVERLAP_LINES).max(chunk_start_idx);
                chunk_min_depth = depth;
                pending_trivia_start = None;
            }
        } else if current_len > MAX_CHUNK_CHARS {
            let split_at = find_soft_break(&lines, chunk_start_idx, i, depth);
            if split_at > chunk_start_idx {
                push_fallback_chunk(&lines, chunk_start_idx, split_at, &mut chunks);
                chunk_start_idx = split_at.saturating_sub(OVERLAP_LINES).max(chunk_start_idx) + 1;
                chunk_min_depth = depth;
                pending_trivia_start = None;
            }
        }

        depth += line.matches('{').count() as i64;
        depth -= line.matches('}').count() as i64;
        chunk_min_depth = chunk_min_depth.min(depth);

        i += 1;
    }

    if chunk_start_idx < lines.len() {
        push_fallback_chunk(&lines, chunk_start_idx, lines.len() - 1, &mut chunks);
    }

    if chunks.is_empty() {
        vec![CodeChunk {
            content: content.to_string(),
            start_line: 1,
            end_line: lines.len(),
        }]
    } else {
        chunks
    }
}

fn find_soft_break(
    lines: &[&str],
    chunk_start: usize,
    hint_idx: usize,
    depth_at_hint: i64,
) -> usize {
    let search_start = chunk_start.max(hint_idx.saturating_sub(15));
    for j in (search_start..=hint_idx).rev() {
        let trimmed = lines[j].trim();
        if trimmed.is_empty() {
            return j;
        }
        if trimmed == "}" && depth_at_hint <= 1 {
            return j;
        }
    }
    hint_idx
}

fn push_fallback_chunk(
    lines: &[&str],
    start_idx: usize,
    end_idx: usize,
    chunks: &mut Vec<CodeChunk>,
) {
    if start_idx > end_idx {
        return;
    }
    let text = lines[start_idx..=end_idx].join("\n");
    if text.trim().is_empty() {
        return;
    }
    chunks.push(CodeChunk {
        content: text,
        start_line: start_idx + 1,
        end_line: end_idx + 1,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_comment_stays_attached_to_its_fn() {
        let src = r#"
/// Adds two numbers together.
/// Returns the sum.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let chunks = chunk_rust(src).expect("should parse");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("Adds two numbers"));
        assert!(chunks[0].content.contains("pub fn add"));
    }

    #[test]
    fn nested_impl_methods_not_confused_with_top_level_items() {
        let src = r#"
struct Foo;

impl Foo {
    fn bar(&self) -> i32 {
        1
    }

    fn baz(&self) -> i32 {
        2
    }
}
"#;
        let chunks = chunk_rust(src).expect("should parse");
        // struct + impl = 2 top-level chunks, not 4 fragmented ones.
        assert_eq!(chunks.len(), 2);
        assert!(chunks[1].content.contains("impl Foo"));
        assert!(chunks[1].content.contains("fn bar"));
        assert!(chunks[1].content.contains("fn baz"));
    }

    #[test]
    fn falls_back_on_syntax_error() {
        let src = "fn broken( {{{ this is not valid rust";
        let chunks = chunk_code(src, "rs");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn generic_fallback_used_for_unsupported_language() {
        let src = (0..80)
            .map(|i| format!("line number {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_code(&src, "shader");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn oversized_impl_splits_per_method_with_header_context() {
        let mut body = String::from("impl Widget {\n");
        for i in 0..40 {
            body.push_str(&format!(
                "    fn method_{i}(&self) -> i32 {{\n        // padding line to grow this method body substantially so the whole impl block exceeds the chunk size budget\n        {i}\n    }}\n\n"
            ));
        }
        body.push_str("}\n");

        let chunks = chunk_rust(&body).expect("should parse");
        assert!(
            chunks.len() > 1,
            "oversized impl should split into multiple chunks"
        );
        assert!(chunks.iter().all(|c| c.content.contains("impl Widget")));
    }

    #[cfg(test)]
    mod cs_tests {
        use super::*;

        #[test]
        fn csharp_attributes_and_comments_stay_attached() {
            let src = r#"
            using UnityEngine;

            public class Player : MonoBehaviour {
                [SerializeField]
                [Header("Movement")]
                private float speed = 10f;

                /// 
                /// Unity Start lifecycle method.
                /// 
                void Start() {
                    Debug.Log("Initialized");
                }
            }
            "#;
            let chunks = chunk_c_sharp(src).expect("should parse");
            assert_eq!(chunks.len(), 1);
            assert!(chunks[0].content.contains("[SerializeField]"));
            assert!(chunks[0].content.contains("void Start()"));
        }

        #[test]
        fn oversized_monobehaviour_splits_per_method_with_class_header() {
            let mut body = String::from("public class EnemyAI : MonoBehaviour {\n");
            for i in 0..30 {
                body.push_str(&format!(
                "    [ClientRpc]\n    void RpcPerformAction_{i}() {{\n        // large padding line to force chunk overflow beyond MAX_CHUNK_CHARS budget\n        Debug.Log(\"Action {i}\");\n    }}\n\n"
            ));
            }
            body.push_str("}\n");

            let chunks = chunk_c_sharp(&body).expect("should parse");
            assert!(chunks.len() > 1, "oversized class should split");
            assert!(
                chunks
                    .iter()
                    .all(|c| c.content.contains("public class EnemyAI : MonoBehaviour"))
            );
            assert!(chunks[0].content.contains("[ClientRpc]"));
        }
    }
}

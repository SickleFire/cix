use tree_sitter::{Parser, Node};
use colored::*;

#[derive(Debug, Clone)]
pub struct RustSymbol {
    pub kind: String,
    pub name: String,
    pub line: usize,
    pub signature: String,
}

pub fn parse_rust_symbols(source_code: &str) -> Vec<RustSymbol> {
    let mut parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    parser
        .set_language(&language)
        .expect("Error loading Rust language");

    let tree: tree_sitter::Tree = match parser.parse(source_code.as_bytes(), None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let mut symbols = Vec::new();
    let root_node = tree.root_node();
    
    walk_node(root_node, source_code, &mut symbols);
    symbols
}

fn walk_node(node: Node, source: &str, symbols: &mut Vec<RustSymbol>) {
    let kind = node.kind();

    match kind {
        "function_item" | "struct_item" | "enum_item" | "trait_item" | "impl_item" | "mod_item" | "macro_definition" => {
            if let Some(symbol) = extract_symbol(node, source) {
                symbols.push(symbol);
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_node(child, source, symbols);
    }
}

fn extract_symbol(node: Node, source: &str) -> Option<RustSymbol> {
    let kind = node.kind().to_string();
    let line = node.start_position().row + 1;
    
    let source_bytes = source.as_bytes();
    let name = match node.child_by_field_name("name") {
        Some(n) => n.utf8_text(source_bytes).unwrap_or("").to_string(),
        None => {
            if node.kind() == "impl_item" {
                if let Some(type_node) = node.child_by_field_name("type") {
                    format!("impl {}", type_node.utf8_text(source_bytes).unwrap_or(""))
                } else {
                    "impl".to_string()
                }
            } else {
                "anonymous".to_string()
            }
        }
    };

    let text = node.utf8_text(source_bytes).unwrap_or("");
    let signature = text.lines().next().unwrap_or(text).trim().to_string();

    Some(RustSymbol {
        kind,
        name,
        line,
        signature,
    })
}

pub fn print_ast(source_code: &str) {
    let mut parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    parser
        .set_language(&language)
        .expect("Error loading Rust language");

    if let Some(tree) = parser.parse(source_code.as_bytes(), None) {
        print_node_recursive(tree.root_node(), source_code, 0);
    }
}

fn print_node_recursive(node: Node, source: &str, depth: usize) {
    let indent = "  ".repeat(depth);
    let kind = node.kind();
    let range = format!("[{}:{} - {}:{}]", 
        node.start_position().row + 1, node.start_position().column,
        node.end_position().row + 1, node.end_position().column
    );

    let text_snippet = if node.child_count() == 0 {
        let text = node.utf8_text(source.as_bytes()).unwrap_or("");
        if text.len() < 30 && !text.contains('\n') {
            format!("  \"{}\"", text)
        } else {
            "".to_string()
        }
    } else {
        "".to_string()
    };

    println!("{}{}{}{}", indent, kind.cyan(), range.dimmed(), text_snippet);

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        print_node_recursive(child, source, depth + 1);
    }
}

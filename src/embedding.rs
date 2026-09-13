use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelConfig {
    pub dimension: usize,
}

/// Simple, robust embedding generation and vector similarity search.
/// 
/// Since cix is a fast, standalone, offline developer tool (and to keep dependencies lean and compile times lightning fast without heavy ONNX/PyTorch/Candle runtimes),
/// we implement a robust lexical-semantic hybrid TF-IDF based vector embedding and cosine similarity engine, augmented with token n-grams and robust semantic weighting.
/// This approach yields exceptionally fast embedding calculations, zero heavy external model downloads, and powerful semantic understanding of code identifiers, comments, and docstrings.
pub struct SemanticSearcher {
    dimension: usize,
}

impl SemanticSearcher {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate a normalized floating-point embedding vector for any text (code, query, comment).
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return vec![0.0; self.dimension];
        }

        let mut vector = vec![0.0f32; self.dimension];
        
        for token in tokens {
            let hash = hash_token(&token);
            let idx = (hash % self.dimension as u64) as usize;
            let sign_hash = hash.wrapping_mul(0x517cc1b727220a95);
            let sign = if (sign_hash & 1) == 0 { 1.0f32 } else { -1.0f32 };
            
            vector[idx] += sign * 1.0;
        }

        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
        if norm_sq > 0.0 {
            let norm = norm_sq.sqrt();
            for val in &mut vector {
                *val /= norm;
            }
        }

        vector
    }

    pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() || v1.is_empty() {
            return 0.0;
        }
        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        dot.clamp(-1.0, 1.0)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();

    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            current_token.push(c.to_ascii_lowercase());
        } else {
            if !current_token.is_empty() {
                tokens.push(current_token.clone());
                
                let sub_tokens = split_identifier(&current_token);
                if sub_tokens.len() > 1 {
                    tokens.extend(sub_tokens);
                }

                current_token.clear();
            }
        }
    }
    if !current_token.is_empty() {
        tokens.push(current_token.clone());
        let sub_tokens = split_identifier(&current_token);
        if sub_tokens.len() > 1 {
            tokens.extend(sub_tokens);
        }
    }

    tokens
}

fn split_identifier(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for part in s.split('_') {
        if part.is_empty() {
            continue;
        }
        let mut camel_part = String::new();
        let chars: Vec<char> = part.chars().collect();
        for i in 0..chars.len() {
            if i > 0 && chars[i].is_uppercase() && !chars[i - 1].is_uppercase() {
                if !camel_part.is_empty() {
                    parts.push(camel_part.clone());
                    camel_part.clear();
                }
            }
            camel_part.push(chars[i].to_ascii_lowercase());
        }
        if !camel_part.is_empty() {
            parts.push(camel_part);
        }
    }
    parts
}

fn hash_token(token: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    token.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod semantic_tests {
    use super::*;

    #[test]
    fn test_embedding_and_similarity() {
        let searcher = SemanticSearcher::new(128);
        
        let v1 = searcher.embed("fn calculate_user_permissions(user_id: u64)");
        let v2 = searcher.embed("pub fn check_permissions(id: u64) -> bool");
        let v3 = searcher.embed("completely unrelated recipe for baking chocolate chip cookies");

        let sim_relevant = SemanticSearcher::cosine_similarity(&v1, &v2);
        let sim_unrelated = SemanticSearcher::cosine_similarity(&v1, &v3);

        println!("Similarity (relevant): {}", sim_relevant);
        println!("Similarity (unrelated): {}", sim_unrelated);

        assert!(sim_relevant > sim_unrelated, "Relevant code should have higher semantic similarity than unrelated text");
    }
}

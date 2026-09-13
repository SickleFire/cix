pub mod parser;
pub mod embedding;

#[cfg(test)]
mod tests {
    use super::parser::{parse_rust_symbols, print_ast};

    const SAMPLE_CODE: &str = r#"
        pub fn hello_world(name: &str) -> String {
            format!("Hello, {}!", name)
        }

        pub struct Greeter {
            prefix: String,
        }

        impl Greeter {
            pub fn greet(&self, name: &str) {
                println!("{} {}", self.prefix, name);
            }
        }
    "#;

    #[test]
    fn test_parse_rust_symbols() {
        let symbols = parse_rust_symbols(SAMPLE_CODE);
        assert!(!symbols.is_empty(), "Symbols should not be empty");

        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"hello_world"), "Should find hello_world function");
        assert!(names.contains(&"Greeter"), "Should find Greeter struct");
        assert!(names.contains(&"impl Greeter"), "Should find impl Greeter block");
        assert!(names.contains(&"greet"), "Should find greet method");

        println!("Parsed symbols: {:#?}", symbols);
    }

    #[test]
    fn test_print_ast() {
        // Just verify print_ast runs without panicking
        print_ast(SAMPLE_CODE);
    }
}

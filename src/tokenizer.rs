#[derive(Debug)]
pub struct Token<'a> {
    pub(crate) s: &'a str,
    pub(crate) start_index: usize,
}

pub fn tokenize(code: &str) -> impl Iterator<Item=Token> {
    let mut next_token = 0;
    let code_length = code.len();
    code.char_indices()
        .filter_map(move |(index, c)|
            if c.is_whitespace() {
                let start_index = next_token;
                next_token = index + 1;
                Some(Token { s: &code[start_index..index], start_index })
            } else if index == code_length - 1 {
                Some(Token { s: &code[next_token..index + 1], start_index: next_token })
            } else {
                None
            })
}
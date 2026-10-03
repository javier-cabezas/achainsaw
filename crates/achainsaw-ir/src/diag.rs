use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub status: String, // "error" or "warning"
    pub error_code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction_index: Option<usize>,
    pub span: Span,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<serde_json::Value>,
}

impl Diagnostic {
    pub fn error(code: impl Into<String>, msg: impl Into<String>, span: Span) -> Self {
        Self {
            status: "error".to_string(),
            error_code: code.into(),
            message: msg.into(),
            instruction_index: None,
            span,
            context: None,
        }
    }

    pub fn with_instruction_index(mut self, idx: usize) -> Self {
        self.instruction_index = Some(idx);
        self
    }

    pub fn with_context(mut self, ctx: serde_json::Value) -> Self {
        self.context = Some(ctx);
        self
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

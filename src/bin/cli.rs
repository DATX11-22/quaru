use std::fmt::Display;

use inquire::{error::InquireError, Select};

enum Choice {
    Show,
    Apply
}

impl Choice {
    fn choices() -> Vec<Choice> {
        vec![Choice::Show, Choice::Apply]
    }
}

impl Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Choice::Apply => write!(f, "Apply"),
            Choice::Show => write!(f, "Show"),
        }
    }
}

enum UnaryOperation {
    Identity,
    Hadamard,
    Phase,
    NOT,
    PauliX,
    PauliY,
    PauliZ,
}

enum BinaryOperation {
    CNOT,
    Swap,
}

/// Prompts the user for an initial choice.
/// 
/// Choices include:
/// - Applying an operation 
/// - Showing the register state
fn initial_prompt() -> Result<Choice, InquireError> {
    let options = Choice::choices(); 
    Select::new("Select an option: ", options).prompt()
}

fn main() {
    let initial = initial_prompt();
}

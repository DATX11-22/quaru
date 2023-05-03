use inquire::{error::InquireError, validator::Validation, Select, Text};
use std::{fmt::Display, vec}; 

/// Initial choices
enum Choice {
    Show,
    Circuit,
    Apply,
    Measure,
    Create,
    OpenQASM,
    Exit,
}

impl Choice {
    /// The possible actions the user can take. Which actions are available depend on the state
    /// of the simulation.
    fn choices(state: &State) -> Vec<Choice> {
        let mut choices = Vec::new();
        if !state.q_regs.is_empty() || !state.c_regs.is_empty() {
            choices.append(&mut vec![Choice::Show]);
        }
        if !state.q_regs.is_empty() {
            choices.append(&mut vec![Choice::Apply, Choice::Measure, Choice::Circuit]);
        }

        choices.append(&mut vec![Choice::Create, Choice::OpenQASM, Choice::Exit]);

        choices
    }
}

impl Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Choice::Apply => write!(f, "Apply"),
            Choice::Measure => write!(f, "Measure"),
            Choice::Show => write!(f, "Show"),
            Choice::Circuit => write!(f, "Show Circuit"),
            Choice::Create => write!(f, "Create register"),
            Choice::OpenQASM => write!(f, "Run OpenQASM program"),
            Choice::Exit => write!(f, "Exit"),
        }
    }
}

#[derive(Debug)]
enum OperationType {
    Unary,
    Binary,
}

impl OperationType {
    fn types() -> Vec<OperationType> {
        vec![OperationType::Unary, OperationType::Binary]
    }

    fn size(&self) -> usize {
        match self {
            OperationType::Unary => 1,
            OperationType::Binary => 2,
        }
    }
}

impl Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::Unary => write!(f, "Unary"),
            OperationType::Binary => write!(f, "Binary"),
        }
    }
}

enum UnaryOperation {
    Identity,
    Hadamard,
    Phase,
    Not,
    PauliY,
    PauliZ,
}

impl UnaryOperation {
    /// Returns a vector of every possible unary operation.
    fn operations() -> Vec<UnaryOperation> {
        vec![
            UnaryOperation::Identity,
            UnaryOperation::Hadamard,
            UnaryOperation::Phase,
            UnaryOperation::Not,
            UnaryOperation::PauliY,
            UnaryOperation::PauliZ,
        ]
    }
}

impl Display for UnaryOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOperation::Identity => write!(f, "Identity"),
            UnaryOperation::Hadamard => write!(f, "Hadamard"),
            UnaryOperation::Phase => write!(f, "Phase"),
            UnaryOperation::Not => write!(f, "NOT"),
            UnaryOperation::PauliY => write!(f, "Pauli Y"),
            UnaryOperation::PauliZ => write!(f, "Pauli Z"),
        }
    }
}

fn unary_operation_target_name(_: &UnaryOperation) -> [&str; 1] {
    ["target"]
}

enum BinaryOperation {
    CNot,
    Swap,
}

impl BinaryOperation {
    /// Returns a vector of every possible binary operation.
    fn operations() -> Vec<BinaryOperation> {
        vec![BinaryOperation::CNot, BinaryOperation::Swap]
    }
}

impl Display for BinaryOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperation::CNot => write!(f, "CNOT"),
            BinaryOperation::Swap => write!(f, "Swap"),
        }
    }
}

fn binary_operation_target_names(op: &BinaryOperation) -> [&str; 2] {
    match *op {
        BinaryOperation::CNot => ["control", "target"],
        _ => ["target"; 2],
    }
}


/// Given a usize `max` prompts the user for a register size and returns a result containing a size.
///
/// # Panics
/// Panics if `max` == 0.
fn size_prompt(max: usize) -> Result<usize, InquireError> {
    assert!(max > 0, "Register size must be atleast 1");

    let options: Vec<usize> = (1..=max).collect();
    Select::new("Select a register size: ", options).prompt()
}

/// Prompts the user for an initial choice and returns the result containing the choice.
///
/// Choices include:
/// - Applying an operation
/// - Showing the register state
/// - Measuring a qubit
/// - Creating a register
/// - Exiting the application
fn init_prompt(state: &State) -> Result<Choice, InquireError> {
    let options = Choice::choices(state);
    Select::new("Select an option: ", options).prompt()
}

/// Given a register size `size` prompts the user for an operation type and returns the result
/// containing the type.
///
/// Types include:
/// - Unary (if `size` >= 1)
/// - Binary (if `size` >= 2)
fn operation_prompt(size: usize) -> Result<OperationType, InquireError> {
    let options = OperationType::types()
        .into_iter()
        .filter(|op_type| op_type.size() <= size)
        .collect();
    Select::new("Select an operation type: ", options).prompt()
}

/// Prompts the user for a unary operation gate and returns the result containing the operation
/// enum.
///
/// Operations include:
/// - Identity
/// - Hadamard
/// - Phase
/// - Not
/// - Pauli Y
/// - Pauli Z
fn unary_prompt() -> Result<UnaryOperation, InquireError> {
    let options = UnaryOperation::operations();
    Select::new("Select an operation: ", options).prompt()
}

/// Prompts the user for a binary operation gate and returns the result containing the operation
/// enum.
///
/// Operations include:
/// - CNot
/// - Swap
fn binary_prompt() -> Result<BinaryOperation, InquireError> {
    let options = BinaryOperation::operations();
    Select::new("Select an operation: ", options).prompt()
}

/// Given an array of target names and a size, prompts the user for `N` selections of indeces
/// from 0 to `size` - 1 and returns the result containing a vector of the selected indeces.
///
/// # Panics
///
/// Panics if `N` is greater than `size`.
fn indicies_prompt<const N: usize>(
    target_names: [&str; N],
    size: usize,
    ) -> Result<Vec<usize>, InquireError> {
    assert!(
        N <= size,
        "Cannot call operation on more qubits than register size! ({N} > {size}"
        );

    let options: Vec<usize> = (0..size).collect();
    let mut targets: Vec<usize> = Vec::new();

    // Prompts the user to select an index for each of the elements in `target_names`
    for name in target_names.iter().take(N) {
        let target = Select::new(
            format!("Select a {name} index: ").as_str(),
            options
            .clone()
            .into_iter()
            .filter(|o| !targets.contains(o))
            .collect(),
            )
            .prompt()?;

        // removes the selected target to avoid duplicate targets
        targets.push(target);
    }

    Ok(targets)
}

#[derive(Debug)]
enum RegisterType {
    Classical,
    Quantum,
}

impl Display for RegisterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

/// Prompts the user for a type of register. Either `Quantum` or `Classical`
fn register_type_prompt() -> Result<RegisterType, InquireError> {
    Select::new(
        "Select the register type",
        vec![RegisterType::Quantum, RegisterType::Classical],
        )
        .prompt()
}

/// Prompts the user for a register name. The name cannot be empty and must
/// not already be used in the supplied `State`.
fn register_name_prompt(state: &State) -> Result<String, InquireError> {
    // Validator that makes sure that the supplied register name is not empty
    let empty_str_validator = |s: &str| {
        if !s.is_empty() {
            Ok(Validation::Valid)
        } else {
            Ok(Validation::Invalid("Register name cannot be empty".into()))
        }
    };

    // Validator that makes sure that the supplied register name is not already
    // used by another register
    let qreg_names: Vec<String> = state.q_regs.keys().cloned().collect();
    let creg_names: Vec<String> = state.q_regs.keys().cloned().collect();
    let no_duplicate_validator = move |s: &str| {
        let s = &s.to_string();
        if qreg_names.contains(s) || creg_names.contains(s) {
            Ok(Validation::Invalid("Register name is already used".into()))
        } else {
            Ok(Validation::Valid)
        }
    };

    // Prompt for register name
    Text::new("Register name: ")
        .with_validator(empty_str_validator)
        .with_validator(no_duplicate_validator)
        .prompt()
}

/// Prompts the user for a register in the specified register collection
///
/// # Arguments
/// * `message` - The message which is displayed during the prompt
/// * `registers` - The collection of registers to choose from
/// * `autoselect` - Whether to autoselect a register if there is only one
fn reg_prompt<T>(
    message: String,
    registers: &mut RegCollection<T>,
    autoselect: bool,
    ) -> Result<&mut T, InquireError> {
    let options: Vec<String> = registers.keys().cloned().collect();

    // Prompt for the quantum register. If there is only one then we don't need to
    // display the prompt.
    let choice = if options.len() == 1 && autoselect {
        options[0].clone()
    } else {
        Select::new(&message, options).prompt()?
    };

    registers
        .get_mut(&choice)
        .ok_or(InquireError::InvalidConfiguration(
                "Invalid quantum register".to_string(),
                ))
}

/// Prompts the user for a quantum register in the specified register collection
fn qreg_prompt(
    registers: &mut QRegCollection,
    autoselect: bool,
    ) -> Result<&mut HistoryQRegister, InquireError> {
    reg_prompt("Select quantum register".to_string(), registers, autoselect)
}

/// Prompts the user for a classical register in the specified register collection
fn creg_prompt(
    registers: &mut CRegCollection,
    autoselect: bool,
    ) -> Result<&mut Vec<bool>, InquireError> {
    reg_prompt(
        "Select classical register".to_string(),
        registers,
        autoselect,
        )
}

/// Like `creg_prompt` but with a custom message
fn creg_prompt_message(
    message: String,
    registers: &mut CRegCollection,
    autoselect: bool,
    ) -> Result<&mut Vec<bool>, InquireError> {
    reg_prompt(message, registers, autoselect)
}

/// Given a register size `size`, prompts the user for a unary operation and a target qubit and
/// returns the result containing an operation.
///
/// # Panics
///
/// Panics if `size` == 0.
fn get_unary(size: usize) -> Result<IdentfiableOperation, InquireError> {
    let unary_op = unary_prompt()?;
    let target = indicies_prompt(unary_operation_target_name(&unary_op), size)?[0];

    let op = match unary_op {
        UnaryOperation::Identity => IdentfiableOperation::identity(target),
        UnaryOperation::Hadamard => IdentfiableOperation::hadamard(target),
        UnaryOperation::Phase => IdentfiableOperation::phase(target),
        UnaryOperation::Not => IdentfiableOperation::not(target),
        UnaryOperation::PauliY => IdentfiableOperation::pauli_y(target),
        UnaryOperation::PauliZ => IdentfiableOperation::pauli_z(target),
    };

    Ok(op)
}

/// Given a register size `size`, prompts the user for a binary operation and a target qubit and
/// returns the result containing an operation.
///
/// # Panics
///
/// Panics if `size` < 2.
fn get_binary(size: usize) -> Result<IdentfiableOperation, InquireError> {
    let binary_op = binary_prompt()?;
    let targets = indicies_prompt(binary_operation_target_names(&binary_op), size)?;

    let a = targets[0];
    let b = targets[1];

    let op = match binary_op {
        BinaryOperation::CNot => IdentfiableOperation::cnot(a, b),
        BinaryOperation::Swap => IdentfiableOperation::swap(a, b),
    };

    Ok(op)
}


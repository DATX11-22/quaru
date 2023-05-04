use inquire::{Select, InquireError, validator::Validation, Text};

use super::{types::{Choice, State, OperationType, UnaryOperation, BinaryOperation, RegisterType, RegCollection, QRegCollection, HistoryQRegister, CRegCollection}, display::IdentfiableOperation};

/// Given a usize `max` prompts the user for a register size and returns a result containing a size.
///
/// # Panics
/// Panics if `max` == 0.
pub fn size_prompt(max: usize) -> Result<usize, InquireError> {
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
pub fn init_prompt(state: &State) -> Result<Choice, InquireError> {
    let options = Choice::choices(state);
    Select::new("Select an option: ", options).prompt()
}

/// Given a register size `size` prompts the user for an operation type and returns the result
/// containing the type.
///
/// Types include:
/// - Unary (if `size` >= 1)
/// - Binary (if `size` >= 2)
pub fn operation_prompt(size: usize) -> Result<OperationType, InquireError> {
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
pub fn unary_prompt() -> Result<UnaryOperation, InquireError> {
    let options = UnaryOperation::operations();
    Select::new("Select an operation: ", options).prompt()
}

/// Prompts the user for a binary operation gate and returns the result containing the operation
/// enum.
///
/// Operations include:
/// - CNot
/// - Swap
pub fn binary_prompt() -> Result<BinaryOperation, InquireError> {
    let options = BinaryOperation::operations();
    Select::new("Select an operation: ", options).prompt()
}

/// Given an array of target names and a size, prompts the user for `N` selections of indeces
/// from 0 to `size` - 1 and returns the result containing a vector of the selected indeces.
///
/// # Panics
///
/// Panics if `N` is greater than `size`.
pub fn indicies_prompt<const N: usize>(
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

/// Prompts the user for a type of register. Either `Quantum` or `Classical`
pub fn register_type_prompt() -> Result<RegisterType, InquireError> {
    Select::new(
        "Select the register type",
        vec![RegisterType::Quantum, RegisterType::Classical],
        )
        .prompt()
}

/// Prompts the user for a register name. The name cannot be empty and must
/// not already be used in the supplied `State`.
pub fn register_name_prompt(state: &State) -> Result<String, InquireError> {
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
    let qreg_names: Vec<String> = state.q_regs().keys().cloned().collect();
    let creg_names: Vec<String> = state.q_regs().keys().cloned().collect();
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
pub fn reg_prompt<T>(
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
pub fn qreg_prompt(
    registers: &mut QRegCollection,
    autoselect: bool,
    ) -> Result<&mut HistoryQRegister, InquireError> {
    reg_prompt("Select quantum register".to_string(), registers, autoselect)
}

/// Prompts the user for a classical register in the specified register collection
pub fn creg_prompt(
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
pub fn creg_prompt_message(
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
pub fn get_unary(size: usize) -> Result<IdentfiableOperation, InquireError> {
    let unary_op = unary_prompt()?;
    let target = indicies_prompt(UnaryOperation::target_name(&unary_op), size)?[0];

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
pub fn get_binary(size: usize) -> Result<IdentfiableOperation, InquireError> {
    let binary_op = binary_prompt()?;
    let targets = indicies_prompt(BinaryOperation::target_names(&binary_op), size)?;

    let a = targets[0];
    let b = targets[1];

    let op = match binary_op {
        BinaryOperation::CNot => IdentfiableOperation::cnot(a, b),
        BinaryOperation::Swap => IdentfiableOperation::swap(a, b),
    };

    Ok(op)
}


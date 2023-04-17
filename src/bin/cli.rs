use clap::Parser;
use inquire::{
    error::InquireError,
    validator::Validation,
    Select, Text,
};

use std::{collections::HashMap, fmt::Display, vec};

use quaru::{
    operation::{self, Operation},
    register::Register,
};

/// Initial choices
enum Choice {
    Show,
    Apply,
    Measure,
    Create,
    Exit,
}

impl Choice {
    fn choices() -> Vec<Choice> {
        vec![
            Choice::Show,
            Choice::Apply,
            Choice::Measure,
            Choice::Create,
            Choice::Exit,
        ]
    }
}

impl Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Choice::Apply => write!(f, "Apply"),
            Choice::Measure => write!(f, "Measure"),
            Choice::Show => write!(f, "Show"),
            Choice::Create => write!(f, "Create"),
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
fn init_prompt() -> Result<Choice, InquireError> {
    let options = Choice::choices();
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
fn qubit_prompt<const N: usize>(
    target_names: [&str; N],
    size: usize,
) -> Result<Vec<usize>, InquireError> {
    assert!(
        N <= size,
        "Cannot call operation on more qubits than register size! ({N} > {size}"
    );

    let options: Vec<usize> = (0..size).collect();
    let mut targets: Vec<usize> = Vec::new();

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
        targets.push(target);
        // removes the selected target to avoid duplicate targets
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
        if s.len() > 0 {
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
        }
        else {
            Ok(Validation::Valid)
        }
    };

    // Prompt for register name
    Text::new("Register name: ")
        .with_validator(empty_str_validator)
        .with_validator(no_duplicate_validator)
        .prompt()
}

/// Prompts the user for a quantum register in the specified `State`
fn qreg_prompt(state: &mut State) -> Result<&mut Register, InquireError> {
    let options: Vec<String> = state.q_regs.keys().cloned().collect();

    // Prompt for the quantum register. If there is only one then we don't need to
    // display the prompt.
    let choice = if options.len() == 1 {
        options[0].clone()
    } else {
        Select::new("Select a quantum register: ", options).prompt()?
    };

    state
        .q_regs
        .get_mut(&choice)
        .ok_or(InquireError::InvalidConfiguration(
            "Invalid quantum register".to_string(),
        ))
}

// /// Prompts the user for a classical register in the specified `State`
// fn creg_prompt(state: &mut State) -> Result<&mut Vec<bool>, InquireError> {
//     let options: Vec<&String> = state.c_regs.keys().collect();
//     let choice = Select::new("Select a classical register: ", options).prompt()?;
//     state
//         .c_regs
//         .get_mut(choice)
//         .ok_or(InquireError::InvalidConfiguration(
//             "Invalid classical register".to_string(),
//         ))
// }

/// Given a register size `size`, prompts the user for a unary operation and a target qubit and
/// returns the result containing an operation.
///
/// # Panics
///
/// Panics if an error occurs during any of the prompts or if `size` == 0.
fn get_unary(size: usize) -> Result<Operation, InquireError> {
    let unary_op = unary_prompt().expect("Problem encountered when selecting unary operation");

    let target = qubit_prompt(unary_operation_target_name(&unary_op), size)
        .expect("Problem encountered when selecting index")[0];

    let op = match unary_op {
        UnaryOperation::Identity => operation::identity(target),
        UnaryOperation::Hadamard => operation::hadamard(target),
        UnaryOperation::Phase => operation::phase(target),
        UnaryOperation::Not => operation::not(target),
        UnaryOperation::PauliY => operation::pauli_y(target),
        UnaryOperation::PauliZ => operation::pauli_z(target),
    };

    Ok(op)
}

/// Given a register size `size`, prompts the user for a binary operation and a target qubit and
/// returns the result containing an operation.
///
/// # Panics
///
/// Panics if an error occurs during any of the prompts or if `size` < 2.
/// TODO: also panics if targets are not in ascending order.
fn get_binary(size: usize) -> Result<Operation, InquireError> {
    let binary_op = binary_prompt().expect("Problem encountered when selecting binary operation");

    let targets = qubit_prompt(binary_operation_target_names(&binary_op), size)
        .expect("Problem encountered when selecting index");

    let a = targets[0];
    let b = targets[1];

    let op = match binary_op {
        BinaryOperation::CNot => operation::cnot(a, b),
        BinaryOperation::Swap => operation::swap(a, b),
    };

    Ok(op)
}

/// Given a mutable simulator state `state` prompts the user for an operation and applies it to a
/// register in the simulator state.
///
/// # Panics
/// Panics if an error occurs while selecting an operation or when the operation is applied.
fn handle_apply(state: &mut State) {
    let reg = qreg_prompt(state).expect("Problem encountered when selecting a quantum register");

    let op_type =
        operation_prompt(reg.size()).expect("Problem encountered during operation type selection");

    let result = match op_type {
        OperationType::Unary => get_unary(reg.size()),
        OperationType::Binary => get_binary(reg.size()),
    };

    match result {
        Ok(op) => reg.apply(&op),
        Err(e) => panic!("Problem encountered when applying operation: {e:?}"),
    };
}

/// Given a mutable simulator state `state` prompts the user for a qubit and measures it, printing the result.
///
/// # Panics
/// Panics if an error occurs while selecting an index.
fn handle_measure(state: &mut State) {
    let reg = qreg_prompt(state).expect("Problem encountered when selecting a quantum register");

    let index = qubit_prompt(["target"], reg.size())
        .expect("Problem encountered when selecting a qubit")[0];

    let result = reg.measure(index);
    println!("Qubit at index {index} measured {result}");
}

/// Given a simulator state `state`, prompts for and prints an overview of a register state.
fn handle_show(state: &mut State) {
    let reg = qreg_prompt(state).expect("Problem encountered when selecting a quantum register");
    println!("Register of size: {}", reg.size());
    reg.print_state();
}

/// Given a simulator state `state`, prompts the user to create a quantum or classical
fn handle_create(state: &mut State) {
    // Promt for register type
    let reg_type =
        register_type_prompt().expect("Problem encountered when selecting a register type");

    // Prompt for register name
    let reg_name = register_name_prompt(state)
        .expect("Problem encountered when entering a register name");

    // Prompt for register size
    let reg_size = size_prompt(4).expect("Problem encountered when selecting register size");
    let reg_state = &[false].repeat(reg_size);

    // Construct register
    match reg_type {
        RegisterType::Classical => {
            state.c_regs.insert(reg_name, reg_state.clone());
        }
        RegisterType::Quantum => {
            let reg = Register::new(reg_state.as_slice());
            state.q_regs.insert(reg_name, reg);
        }
    }
}

/// A cli-based ideal quantum computer simulator
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of qubits in the quantum register
    #[arg(short, long)]
    size: Option<usize>,
}

/// The state of the simulator
struct State {
    q_regs: HashMap<String, Register>,
    c_regs: HashMap<String, Vec<bool>>,
}

/// Runs the Quaru shell.
fn main() {
    let args = Args::parse();

    println!("{QUARU}");

    // Initialize the state of the simulator
    let mut state = State {
        q_regs: HashMap::new(),
        c_regs: HashMap::new(),
    };

    // Size arg is optional.
    let n = if let Some(size) = args.size {
        size
    } else {
        // 4 max gives a nice wrapping, argument allows for bigger
        size_prompt(4).expect("Problem when selecting a register size")
    };

    // Create initial register
    let init_state = &[false].repeat(n);
    let reg = Register::new(init_state.as_slice());
    state.q_regs.insert("qreg0".to_string(), reg);

    // Clear terminal
    print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

    loop {
        let init = init_prompt().expect("Problem selecting an option");

        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

        match init {
            Choice::Show => handle_show(&mut state),
            Choice::Apply => handle_apply(&mut state),
            Choice::Measure => handle_measure(&mut state),
            Choice::Create => handle_create(&mut state),
            Choice::Exit => break,
        };
    }
}

const QUARU: &str = "
  ______      __    __       ___      .______       __    __  
 /  __  \\    |  |  |  |     /   \\     |   _  \\     |  |  |  | 
|  |  |  |   |  |  |  |    /  ^  \\    |  |_)  |    |  |  |  | 
|  |  |  |   |  |  |  |   /  /_\\  \\   |      /     |  |  |  | 
|  `--'  '--.|  `--'  |  /  _____  \\  |  |\\  \\----.|  `--'  | 
 \\_____\\_____\\\\______/  /__/     \\__\\ | _| `._____| \\______/  
                                                              
";

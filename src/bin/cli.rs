use clap::Parser;
use inquire::{error::InquireError, Select};

use std::{fmt::Display, vec};

use quant::{
    operation::{self, Operation},
    register::Register,
};

/// Initial choices
enum Choice {
    Show,
    Apply,
    Measure,
    Exit,
}

impl Choice {
    fn choices() -> Vec<Choice> {
        vec![Choice::Show, Choice::Apply, Choice::Measure, Choice::Exit]
    }
}

impl Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Choice::Apply => write!(f, "Apply"),
            Choice::Measure => write!(f, "Measure"),
            Choice::Show => write!(f, "Show"),
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
    NOT,
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
            UnaryOperation::NOT,
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
            UnaryOperation::NOT => write!(f, "NOT"),
            UnaryOperation::PauliY => write!(f, "Pauli Y"),
            UnaryOperation::PauliZ => write!(f, "Pauli Z"),
        }
    }
}

enum BinaryOperation {
    CNOT,
    Swap,
}

impl BinaryOperation {
    /// Returns a vector of every possible binary operation.
    fn operations() -> Vec<BinaryOperation> {
        vec![BinaryOperation::CNOT, BinaryOperation::Swap]
    }
}

impl Display for BinaryOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperation::CNOT => write!(f, "CNOT"),
            BinaryOperation::Swap => write!(f, "Swap"),
        }
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
/// - NOT
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
/// - CNOT
/// - Swap
fn binary_prompt() -> Result<BinaryOperation, InquireError> {
    let options = BinaryOperation::operations();
    Select::new("Select an operation: ", options).prompt()
}

/// Given a number of operands `n` and a register size `size`, prompts the user for `n` selections
/// of indeces from 0 to `size` - 1 and returns the result containing a vector of the selected
/// indeces.
///
/// # Panics
///
/// Panics if `n` is greater than `size`.
fn qubit_prompt(n: usize, size: usize) -> Result<Vec<usize>, InquireError> {
    assert!(
        n <= size,
        "Cannot call operation on more qubits than register size! ({n} > {size}"
    );

    let options: Vec<usize> = (0..size).collect();
    let mut targets: Vec<usize> = Vec::new();

    for _ in 0..n {
        let target = Select::new(
            "Select a target index: ",
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

/// Given a register size `size`, prompts the user for a unary operation and a target qubit and
/// returns the result containing an operation.
///
/// # Panics
///
/// Panics if an error occurs during any of the prompts or if `size` == 0.
fn get_unary(size: usize) -> Result<Operation, InquireError> {
    let unary_op = match unary_prompt() {
        Ok(op) => op,
        Err(e) => panic!(
            "Problem encountered when selecting unary operation: {:?}",
            e
        ),
    };

    let target = match qubit_prompt(1, size) {
        Ok(ts) => ts[0],
        Err(e) => panic!("Problem encountered when selecting index: {:?}", e),
    };

    let op = match unary_op {
        UnaryOperation::Identity => operation::identity(target),
        UnaryOperation::Hadamard => operation::hadamard(target),
        UnaryOperation::Phase => operation::phase(target),
        UnaryOperation::NOT => operation::not(target),
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
    let binary_op = match binary_prompt() {
        Ok(op) => op,
        Err(e) => panic!(
            "Problem encountered when selecting binary operation: {:?}",
            e
        ),
    };

    let mut targets = match qubit_prompt(2, size) {
        Ok(ts) => ts,
        Err(e) => panic!("Problem encountered when selecting index: {:?}", e),
    };

    if targets[1] < targets[0] {
        targets.swap(1, 0);
        println!(
            "Descending targets are currently not supported! Swapping {} and {}",
            targets[1], targets[0]
        );
    }

    let op = match binary_op {
        BinaryOperation::CNOT => operation::cnot(targets[1], targets[0]),
        BinaryOperation::Swap => operation::swap(targets[1], targets[0]),
    };

    Ok(op)
}

/// Given a mutable register `reg` prompts the user for an operation and applies it to the
/// register.
///
/// # Panics
/// Panics if an error occurs while selecting an operation or when the operation is applied.
fn handle_apply(reg: &mut Register) {
    let op_type = match operation_prompt(reg.size()) {
        Ok(op_type) => op_type,
        Err(e) => panic!(
            "Problem encountered during operation type selection: {:?}",
            e
        ),
    };

    let result = match op_type {
        OperationType::Unary => get_unary(reg.size()),
        OperationType::Binary => get_binary(reg.size()),
    };

    match result {
        Ok(op) => reg.apply(&op),
        Err(e) => panic!("Problem encountered when applying operation: {:?}", e),
    };
}

/// Given a mutable register `reg` prompts the user for an index and measures the qubit at that
/// index, printing the result.
///
/// # Panics
/// Panics if an error occurs while selecting an index.
fn handle_measure(reg: &mut Register) {
    let index = match qubit_prompt(1, reg.size()) {
        Ok(ts) => ts[0],
        Err(e) => panic!("Problem encountered when selecting a qubit: {:?}", e),
    };
    let result = reg.measure(index.clone());
    println!("Qubit at index {index} measured {result}");
}

/// Given a register `reg` prints an overview of the register state.
fn handle_show(reg: &Register) {
    println!("Register of size: {}", reg.size());
    reg.print_state();
}

/// A cli-based ideal quantum computer simulator
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of qubits in the quantum register
    #[arg(short, long)]
    size: Option<usize>,
}

/// Runs the Quaru shell.
fn main() {
    let args = Args::parse();

    // size arg is optional
    let size = if let Some(n) = args.size {
        n
    } else {
        // 4 max gives a nice wrapping, argument allows for bigger
        match size_prompt(4) {
            Ok(size) => size,
            Err(e) => panic!("Problem when selecting a register size: {:?}", e),
        }
    };

    let init_state = &[false].repeat(size);
    let mut reg = Register::new(init_state.as_slice());
    

    // clear terminal
    print!("{esc}[2J{esc}[1;1H", esc = 27 as char);


    loop {
        println!("{}", QUARU);

        let init = match init_prompt() {
            Ok(choice) => choice,
            Err(e) => panic!("Problem selecting an option: {:?}", e),
        };

        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

        match init {
            Choice::Show => handle_show(&reg),
            Choice::Apply => handle_apply(&mut reg),
            Choice::Measure => handle_measure(&mut reg),
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


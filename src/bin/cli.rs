use clap::Parser;
use inquire::{error::InquireError, validator::Validation, Select, Text};
use std::{collections::HashMap, fmt::Display, path::Path, vec}; 

use quaru::{
    openqasm,
    register::Register,
};

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

/// Given a mutable simulator state `state` prompts the user for an operation and applies it to a
/// register in the simulator state.
///
/// # Panic
///
/// Panics if there are no quantum registers in `state`
fn handle_apply(state: &mut State) -> Result<(), InquireError> {
    let reg = qreg_prompt(&mut state.q_regs, true)?;
    let op_type = operation_prompt(reg.size())?;

    let op = match op_type {
        OperationType::Unary => get_unary(reg.size()),
        OperationType::Binary => get_binary(reg.size()),
    }?;

    reg.apply(&op);

    Ok(())
}

/// Given a mutable simulator state `state` prompts the user for a qubit and measures it, printing the result.
///
/// # Panic
///
/// Panics if there are no quantum registers in `state`
fn handle_measure(state: &mut State) -> Result<(), InquireError> {
    let q_reg = qreg_prompt(&mut state.q_regs, true)?;
    let q_index = indicies_prompt(["qubit"], q_reg.size())?[0];

    let result = q_reg.measure(q_index);

    // Prompt for classical register to prompt into, don't autoselect because the
    // user might not always want to measure into a classical register
    if let Ok(c_reg) = creg_prompt_message(
        "Select classical register (Esc to skip)".to_string(),
        &mut state.c_regs,
        false,
    ) {
        if let Ok(c_index) = indicies_prompt(["bit"], c_reg.len()) {
            c_reg[c_index[0]] = result;
        }
    }

    println!("Qubit at measured {result}");

    Ok(())
}

/// Given a simulator state `state`, prompts for and prints an overview of a register state.
///
/// # Panic
///
/// Panics if there are no registers in `state`
fn handle_show(state: &mut State) -> Result<(), InquireError> {
    // Prompt for the type of register to show, or default to one if the other is empty
    let reg_type = if state.q_regs.is_empty() {
        RegisterType::Classical
    } else if state.c_regs.is_empty() {
        RegisterType::Quantum
    } else {
        register_type_prompt()?
    };

    // Print the registers state
    match reg_type {
        RegisterType::Classical => {
            let reg = creg_prompt(&mut state.c_regs, true)?;

            println!("Classical register of size: {}", reg.len());
            println!("{:?}", reg);
        }
        RegisterType::Quantum => {
            let reg = qreg_prompt(&mut state.q_regs, true)?;

            println!("Quantum register of size: {}", reg.size());
            reg.print_state();
        }
    };

    Ok(())
}

fn handle_circuit(state: &mut State) -> Result<(), InquireError> {
    let register = qreg_prompt(&mut state.q_regs, true)?;
    display_circuit(register.history(), register.size());

    Ok(())
}

/// Given a simulator state `state`, prompts the user to create a quantum or classical
fn handle_create(state: &mut State) -> Result<(), InquireError> {
    // Promt for register type
    let reg_type = register_type_prompt()?;

    // Prompt for register name
    let reg_name = register_name_prompt(state)?;

    // Prompt for register size
    let reg_size = size_prompt(4)?;
    let reg_state = &[false].repeat(reg_size);

    // Construct register
    match reg_type {
        RegisterType::Classical => {
            state.c_regs.insert(reg_name, reg_state.clone());
        }
        RegisterType::Quantum => {
            let reg = HistoryQRegister::new(reg_state.as_slice());
            state.q_regs.insert(reg_name, reg);
        }
    };

    Ok(())
}

/// Given a simulator state `state`, prompts the user for a file containing OpenQASM code, runs the code
/// and updates `state` accordingly. The OpenQASM program is validated before running and an error is display
/// if the specified OpenQASM program is invalid.
fn handle_openqasm(state: &mut State) -> Result<(), InquireError> {
    // Validator to make sure that the supplied filepath links to a valid
    // OpenQASM program.
    let openqasm_validator = |s: &str| {
        if s.is_empty() {
            return Ok(Validation::Valid);
        }

        let path = Path::new(s);
        match openqasm::run_openqasm(path) {
            Ok(_) => Ok(Validation::Valid),
            Err(e) => Ok(Validation::Invalid(
                format!("Parsing error: {:?}", e).into(),
            )),
        }
    };

    // Prompt the user for a filepath to an OpenQASM program, makes sure that
    // the filepath leads to a valid OpenQASM program before returning (Or that the
    // filepath is empty, in which case we should cancel the operation).
    let filepath = Text::new("OpenQASM file path:")
        .with_validator(openqasm_validator)
        .prompt()?;

    // Cancel the operation if the user supplied an empty filepath
    if filepath.is_empty() {
        return Err(InquireError::OperationCanceled);
    }

    // Run the openqasm program, the file should already be validated so this should not
    // panic
    let res = openqasm::run_openqasm(Path::new(&filepath))
        .expect("Problem encountered when running OpenQASM program");

    // Convert quantum registers to historical registers
    let qregs: HashMap<String, HistoryQRegister> = res.qregs.into_iter().map(|(k,v)| (k, HistoryQRegister::from_register(v))).collect();

    // Add the result from the openqasm file to the state of the simulation
    state.q_regs.extend(qregs);
    state.c_regs.extend(res.cregs);

    Ok(())
}

/// A cli-based ideal quantum computer simulator
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of qubits in the quantum register
    #[arg(short, long)]
    size: Option<usize>,
}

use quaru::cli_utils::display::{display_circuit, IdentfiableOperation};

type RegCollection<T> = HashMap<String, T>;
type QRegCollection = RegCollection<HistoryQRegister>;
type CRegCollection = RegCollection<Vec<bool>>;

/// The state of the simulator
struct State {
    q_regs: QRegCollection,
    c_regs: CRegCollection,
}

struct HistoryQRegister {
    register: Register,
    history: Vec<IdentfiableOperation>,
}

impl HistoryQRegister  {
    fn new(input_bits: &[bool]) -> HistoryQRegister {
        let register = Register::new(input_bits);
        let history: Vec<IdentfiableOperation> = Vec::new();

        HistoryQRegister { register, history }
    }

    fn size(&self) -> usize {
        self.register.size()
    }

    fn history(&self) -> Vec<IdentfiableOperation> {
        self.history.clone()
    }

    fn print_state(&self) {
        self.register.print_state()
    }

    fn apply(&mut self, op: &IdentfiableOperation) {
        self.history.push(op.clone());
        self.register.apply(&op.operation());
    }

    fn from_register(register: Register) -> HistoryQRegister {
        HistoryQRegister { register, history: Vec::new() }
    }

    fn measure(&mut self, index: usize) -> bool {
        self.register.measure(index)
    }
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

    // Size arg is optional. Create quantum register on startup if size arg is supplied.
    if let Some(n) = args.size {
        // Create initial register
        let init_state = &[false].repeat(n);
        let reg = HistoryQRegister::new(init_state.as_slice());
        state.q_regs.insert("qreg0".to_string(), reg);
    }

    // Clear terminal
    print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

    loop {
        let init = match init_prompt(&state) {
            Ok(init) => init,
            Err(InquireError::OperationCanceled) => continue,
            Err(e) => panic!("Error occured: {e}"),
        };

        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

        let res = match init {
            Choice::Show => handle_show(&mut state),
            Choice::Circuit => handle_circuit(&mut state),
            Choice::Apply => handle_apply(&mut state),
            Choice::Measure => handle_measure(&mut state),
            Choice::Create => handle_create(&mut state),
            Choice::OpenQASM => handle_openqasm(&mut state),
            Choice::Exit => break,
        };

        match res {
            Ok(_) => {}
            Err(InquireError::OperationCanceled) => {}
            Err(e) => panic!("Error occured: {e}"),
        }
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

use clap::Parser;
use inquire::{error::InquireError, validator::Validation, Text};
use openqasm_parser::openqasm::BasicOp;
use std::{collections::HashMap, path::Path};

use quaru::{
    cli_utils::{
        display::IdentfiableOperation,
        prompt::{
            creg_prompt, creg_prompt_message, get_binary, get_unary, indicies_prompt, init_prompt,
            operation_prompt, qreg_prompt, register_name_prompt, register_type_prompt, size_prompt,
        },
        types::{Choice, HistoryQRegister, OperationType, RegisterType, State},
    },
    openqasm,
    register::Register,
};

/// Given a mutable simulator state `state` prompts the user for an operation and applies it to a
/// register in the simulator state.
///
/// # Panic
///
/// Panics if there are no quantum registers in `state`
fn handle_apply(state: &mut State) -> Result<(), InquireError> {
    let reg = qreg_prompt(state.q_regs_as_mut(), true)?;
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
    let q_reg = qreg_prompt(state.q_regs_as_mut(), true)?;
    let q_index = indicies_prompt(["qubit"], q_reg.size())?[0];

    let result = q_reg.measure(q_index);

    // Prompt for classical register to prompt into, don't autoselect because the
    // user might not always want to measure into a classical register
    if let Ok(c_reg) = creg_prompt_message(
        "Select classical register (Esc to skip)".to_string(),
        &mut state.c_regs(),
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
    let reg_type = if state.q_regs().is_empty() {
        RegisterType::Classical
    } else if state.c_regs().is_empty() {
        RegisterType::Quantum
    } else {
        register_type_prompt()?
    };

    // Print the registers state
    match reg_type {
        RegisterType::Classical => {
            let reg = creg_prompt(state.c_regs_as_mut(), true)?;

            println!("Classical register of size: {}", reg.len());
            println!("{reg:?}");
        }
        RegisterType::Quantum => {
            let reg = qreg_prompt(state.q_regs_as_mut(), true)?;

            println!("Quantum register of size: {}", reg.size());
            reg.print_state();
        }
    };

    Ok(())
}

fn handle_circuit(state: &mut State) -> Result<(), InquireError> {
    let register = qreg_prompt(state.q_regs_as_mut(), true)?;
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
            state.insert_c_reg(reg_name, reg_state.clone());
        }
        RegisterType::Quantum => {
            let reg = HistoryQRegister::new(reg_state.as_slice());
            state.insert_q_reg(reg_name, reg);
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
            Err(e) => Ok(Validation::Invalid(format!("Parsing error: {e:?}").into())),
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

    // Compute the list of operations for the OpenQASM program
    let openqasm_ops: Vec<_> = res
        .program
        .get_basic_operations()
        .into_iter()
        .map(|op| op.1)
        .collect();

    // Convert quantum registers to historical registers
    let qregs: HashMap<String, HistoryQRegister> = res
        .qregs
        .into_iter()
        .map(|(k, v)| (k.clone(), openqasm_history_qreg(&k, v, &openqasm_ops)))
        .collect();

    // Add the result from the openqasm file to the state of the simulation
    state.extend_q_regs(qregs);
    state.extend_c_regs(res.cregs);

    Ok(())
}

fn openqasm_history_qreg(
    register_name: &String,
    register: Register,
    openqasm_operations: &Vec<BasicOp>,
) -> HistoryQRegister {
    let history: Vec<_> = openqasm_operations
        .iter()
        .filter(|op| {
            // Filter out to only include gates applied on the current register
            match op {
                BasicOp::U(_, _, _, q) => &q.0 == register_name,
                BasicOp::CX(q1, q2) => &q1.0 == register_name && &q2.0 == register_name,
                BasicOp::Measure(q, _) => &q.0 == register_name,
                BasicOp::ResetQ(q) => &q.0 == register_name,
                BasicOp::ResetC(_) => false,
            }
        })
        .filter_map(|op| {
            // Map BasicOp -> Identifiable operation
            match op {
                BasicOp::U(theta, phi, lambda, qubit) => {
                    Some(IdentfiableOperation::u(*theta, *phi, *lambda, qubit.1))
                }
                BasicOp::CX(q1, q2) => Some(IdentfiableOperation::cnot(q1.1, q2.1)),
                BasicOp::Measure(q, _) => Some(IdentfiableOperation::measure(q.1)),
                BasicOp::ResetQ(q) => Some(IdentfiableOperation::reset(q.1)),
                BasicOp::ResetC(_) => None,
            }
        })
        .collect();

    HistoryQRegister::from_register(register, history)
}

/// A cli-based ideal quantum computer simulator
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of qubits in the quantum register
    #[arg(short, long)]
    size: Option<usize>,
}

use quaru::cli_utils::display::display_circuit;

/// Runs the Quaru shell.
fn main() {
    let args = Args::parse();

    println!("{QUARU}");

    // Initialize the state of the simulator
    let mut state = State::new();

    // Size arg is optional. Create quantum register on startup if size arg is supplied.
    if let Some(n) = args.size {
        // Create initial register
        let init_state = &[false].repeat(n);
        let reg = HistoryQRegister::new(init_state.as_slice());
        state.insert_q_reg("qreg0".to_string(), reg);
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

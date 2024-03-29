//! Code related to running OpenQASM programs on the simulator.

use crate::{
    math::c64,
    operation::{self, Operation},
    register::Register,
};
use ndarray::array;
use openqasm_parser::openqasm::{self, BasicOp, OpenQASMError as OpenQASMParseError, semantic_analysis::OpenQASMProgram};
use std::{collections::HashMap, path::Path};

/// The result of running an OpenQASM program
pub struct OpenQASMResult {
    /// `qregs` and `cregs` store the final results of the quantum and classical registers after running
    /// the OpenQASM program.
    pub qregs: HashMap<String, Register>,
    pub cregs: HashMap<String, Vec<bool>>,

    /// Contains information about the OpenQASM program
    pub program: OpenQASMProgram,
}

/// The different types of errors that can occur when running an OpenQASM program.
#[derive(Debug)]
pub enum OpenQASMError {
    /// Error when parsing the OpenQASM file. The problem can be with reading the file,
    /// parsing the tokens, syntax or semantics.
    OpenQASMParseError(OpenQASMParseError),

    /// Error when running the program on the simulator.
    OpenQASMRunError(String),
}

/// Reads a file containing OpenQASM 2.0 code and runs it on the simulator.
///
/// If successful: Returns the quanum and classical [Registers] defined in the OpenQASM file in the state
/// they are in after running the program.
///
/// If unsuccessful: Returns an [OpenQASMError].
///
/// The way the simulator is implemented it is not possible to apply a CX gate to qubits in
/// different registers. If the OpenQASM program tries to do this an error is returned.
///
/// See <https://github.com/openqasm/openqasm/tree/OpenQASM2.x> for more information on OpenQASM 2.0.
///
/// # Examples
/// ```
/// use quaru::openqasm;
/// use std::path::Path;
/// let registers = openqasm::run_openqasm(Path::new("filepath.qasm"));
/// ```
pub fn run_openqasm(openqasm_file: &Path) -> Result<OpenQASMResult, OpenQASMError> {
    let program =
        openqasm::parse_openqasm(openqasm_file).map_err(OpenQASMError::OpenQASMParseError)?;

    // Initializes the registers defined in the openqasm file. All registers are initialized
    // to 0.
    let mut result = OpenQASMResult {
        qregs: program
            .qregs
            .iter()
            .map(|(name, &size)| (name.clone(), Register::new(&vec![false; size])))
            .collect(),
        cregs: program
            .cregs
            .iter()
            .map(|(name, &size)| (name.clone(), vec![false; size]))
            .collect(),
        program,
    };

    // Applies the operations defined in the openqasm file to the registers.
    for (condition, op) in result.program.get_basic_operations() {
        // If there is a condition and it is not satisfied, don't apply the operation.
        if let Some(condition) = condition {
            // Assuming the openqasm parser is correctly implemented this should always
            // return a register and never panic.
            let creg = result
                .cregs
                .get(&condition.0)
                .expect("Register does not exist?");

            if creg_to_u32(creg) != condition.1 {
                continue;
            }
        }

        // There was no invalid condition so we apply the operation to some register(s)
        // Calling .expect() on the registers should be safe here assuming the openqasm implementation is
        // correct. If the register didn't exist it should already have returned a SemanticError.
        match op {
            BasicOp::U(p1, p2, p3, q) => {
                let qreg = result
                    .qregs
                    .get_mut(&q.0)
                    .expect("Register does not exist?");
                qreg.apply(
                    &u(p1 as f64, p2 as f64, p3 as f64, q.1).expect("Could not create U operation"),
                );
            }
            BasicOp::CX(q1, q2) => {
                if q1.0 != q2.0 {
                    return Err(OpenQASMError::OpenQASMRunError("The simulator currently does not support applying CX on qubits in two different registers".to_string()));
                }

                let qreg = result
                    .qregs
                    .get_mut(&q1.0)
                    .expect("Register does not exist?");
                qreg.apply(&operation::cnot(q1.1, q2.1));
            }
            BasicOp::Measure(q, c) => {
                let qreg = result
                    .qregs
                    .get_mut(&q.0)
                    .expect("Register does not exist?");
                let creg = result
                    .cregs
                    .get_mut(&c.0)
                    .expect("Register does not exist?");

                creg[c.1] = qreg.measure(q.1);
            }
            BasicOp::ResetQ(q) => {
                let qreg = result
                    .qregs
                    .get_mut(&q.0)
                    .expect("Register does not exist?");
                *qreg = Register::new(&vec![false; qreg.size()]);
            }
            BasicOp::ResetC(c) => {
                let creg = result
                    .cregs
                    .get_mut(&c.0)
                    .expect("Register does not exist?");
                *creg = vec![false; creg.len()];
            }
        };
    }

    Ok(result)
}

/// Returns a universal operation for the given angles on the `target` qubit.
pub fn u(theta: f64, phi: f64, lambda: f64, target: usize) -> Option<Operation> {
    let theta = c64::from(theta);
    let phi = c64::from(phi);
    let lambda = c64::from(lambda);
    let i = c64::i();
    Operation::new(
        array![
            [
                (-i * (phi + lambda) / 2.0).exp() * (theta / 2.0).cos(),
                -(-i * (phi - lambda) / 2.0).exp() * (theta / 2.0).sin()
            ],
            [
                (i * (phi - lambda) / 2.0).exp() * (theta / 2.0).sin(),
                (i * (phi + lambda) / 2.0).exp() * (theta / 2.0).cos()
            ],
        ],
        vec![target],
    )
}

/// Converts a classical register to a u32
///
/// The register is represented by a vector of booleans. This function converts
/// the classical register from this representation into its binary representation
/// where `creg[0]` is the least significant bit (in occordance with the OpenQASM 2.0 spec).
///
/// # Panics
/// Because the return type is a u32 this function panics when given a register larger than
/// 32 bits. This should usually not be a problem as simulating a 32 bit quantum computer is
/// infeasable.
fn creg_to_u32(creg: &Vec<bool>) -> u32 {
    assert!(
        creg.len() <= 32,
        "Registers larger than 32 bits are not allowed"
    );
    let mut res = 0;

    for bit in creg.iter().rev() {
        res <<= 1;
        if *bit {
            res |= 1;
        }
    }

    res
}

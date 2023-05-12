use crate::{
    openqasm,
    operation::{self, Operation, QuantumOperation},
};

/// Displays a list of operations as a circuit diagram
/// Each horizontal line represents a qubit
/// Each vertical line represents an operation
/// Takes a list of operations and the number of qubits in the circuit
pub fn display_circuit(operations: Vec<IdentfiableOperation>, size: usize) {
    let mut circuit: Vec<String> = Vec::new();

    // Push strings to circuit and initialize them with qubit numbers
    // Intersperse each line with an empty line
    for _ in 0..size + (size - 1) {
        circuit.push(String::new());
    }
    for (i, q) in circuit.iter_mut().enumerate() {
        if i % 2 == 1 {
            q.push_str("    ");
        } else {
            q.push_str(&format!("|q{}>", i/2));
        }
    }

    // Go through each operation and match it to the right qubit
    // The qubits unaffected by the operation are left with a "―"

    for idop in operations {
        // Determine binary or unary operation
        if idop.operation.targets().len() != 1 {
            // Binary operation
            let target1 = idop.operation.targets()[0];
            let target2 = idop.operation.targets()[1];
            // Match what operation it is, then write correct symbols
            match idop.identifier {
                OperationIdentifier::CNot => {
                    // Write CNOT symbol
                    circuit[target1 * 2].push('X');
                    circuit[target2 * 2].push('O'); // Control
                }
                OperationIdentifier::Swap => {
                    // Write Swap symbol
                    circuit[target1 * 2].push('X');
                    circuit[target2 * 2].push('X');
                }
                _ => {}
            }

            for (i, q) in circuit.iter_mut().enumerate() {
                if i == target1*2 || i == target2*2 {
                    continue;
                }
                if (target1*2..target2*2).contains(&i) || (target2*2..target1*2).contains(&i) {
                    q.push('|');
                } else if i % 2 == 0 {
                    q.push('―');
                } else {
                    q.push(' ');
                }
            }
        } else {
            // Unary operation
            let target = idop.operation.targets()[0];
            // Match what operation it is, then write correct symbol
            match idop.identifier {
                OperationIdentifier::Identity => {
                    // Write Identity symbol
                    circuit[target * 2].push('I');
                }
                OperationIdentifier::Hadamard => {
                    // Write Hadamard symbol
                    circuit[target * 2].push('H');
                }
                OperationIdentifier::Phase => {
                    // Write Phase symbol
                    circuit[target * 2].push('P');
                }
                OperationIdentifier::Not => {
                    // Write Not symbol
                    circuit[target * 2].push('X');
                }
                OperationIdentifier::PauliY => {
                    // Write Pauli Y symbol
                    circuit[target * 2].push('Y');
                }
                OperationIdentifier::PauliZ => {
                    // Write Pauli Z symbol
                    circuit[target * 2].push('Z');
                }
                OperationIdentifier::U => {
                    // Write OpenQASM U gate symbol
                    circuit[target * 2].push('U');
                }
                OperationIdentifier::Measure => {
                    // Write OpenQASM measure symbol
                    circuit[target * 2].push('/');
                }
                OperationIdentifier::Reset => {
                    // Write OpenQASM reset symbol
                    circuit[target * 2].push('[');
                }
                _ => {}
            }
            for (i, q) in circuit.iter_mut().enumerate() {
                if i == target * 2 {
                    continue;
                }
                if i % 2 == 0 {
                    q.push('―');
                } else {
                    q.push(' ');
                }
            }
        }
    }
    circuit.iter().for_each(|l| println!("{l}"));
}

/// An operation with an identifier.
#[derive(Clone, Debug)]
pub struct IdentfiableOperation {
    identifier: OperationIdentifier,
    operation: Operation,
}

impl IdentfiableOperation {
    fn new(identifier: OperationIdentifier, operation: Operation) -> IdentfiableOperation {
        IdentfiableOperation {
            identifier,
            operation,
        }
    }
    /// Given a target returns an identifiable Identity operation.
    pub fn identity(target: usize) -> IdentfiableOperation {
        Self::new(OperationIdentifier::Identity, operation::identity(target))
    }

    /// Given a target returns an identifiable Hadamard operation.
    pub fn hadamard(target: usize) -> IdentfiableOperation {
        Self::new(OperationIdentifier::Hadamard, operation::hadamard(target))
    }

    /// Given a target returns an identifiable Phase operation.
    pub fn phase(target: usize) -> IdentfiableOperation {
        Self::new(OperationIdentifier::Phase, operation::phase(target))
    }

    /// Given a target returns an identifiable NOT operation.
    pub fn not(target: usize) -> IdentfiableOperation {
        Self::new(OperationIdentifier::Not, operation::not(target))
    }

    /// Given a target returns an identifiable Pauli Y operation.
    pub fn pauli_y(target: usize) -> IdentfiableOperation {
        Self::new(OperationIdentifier::PauliY, operation::pauli_y(target))
    }

    /// Given a target returns an identifiable Pauli Z operation.
    pub fn pauli_z(target: usize) -> IdentfiableOperation {
        Self::new(OperationIdentifier::PauliZ, operation::pauli_z(target))
    }

    /// Given a target returns an identifiable OpenQASM 2.0 U operation.
    pub fn u(theta: f32, phi: f32, lambda: f32, qubit: usize) -> IdentfiableOperation {
        Self::new(
            OperationIdentifier::U,
            openqasm::u(theta as f64, phi as f64, lambda as f64, qubit)
                .expect("Could not construct U gate"),
        )
    }

    /// Given a control and a target returns an identifiable CNOT operation.
    pub fn cnot(control: usize, target: usize) -> IdentfiableOperation {
        Self::new(OperationIdentifier::CNot, operation::cnot(control, target))
    }

    /// Given two targets returns an identifiable Swap operation.
    pub fn swap(target_1: usize, target_2: usize) -> IdentfiableOperation {
        Self::new(
            OperationIdentifier::Swap,
            operation::swap(target_1, target_2),
        )
    }

    pub fn measure(qubit: usize) -> IdentfiableOperation {
        Self::new(OperationIdentifier::Measure, operation::identity(qubit))
    }

    pub fn reset(qubit: usize) -> IdentfiableOperation {
        Self::new(OperationIdentifier::Reset, operation::identity(qubit))
    }

    /// Returns the operation.
    pub fn operation(&self) -> Operation {
        self.operation.clone()
    }
}

/// Identifier for available operations in the CLI.
#[derive(Clone, Debug)]
pub enum OperationIdentifier {
    /// The Identity identifier.
    Identity,
    /// The Hadamard identifier.
    Hadamard,
    /// The Phase identifier.
    Phase,
    /// The Not identifier.
    Not,
    /// The Pauli Y identifier.
    PauliY,
    /// The Pauli Z identifier.
    PauliZ,
    /// The OpenQASM U gate identifier.
    U,
    /// The CNOT identifier.
    CNot,
    /// The Swap identifier.
    Swap,
    /// The measure identifier.
    Measure,
    /// The reset identifier.
    Reset,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_display_circuit() {
        let mut operations: Vec<IdentfiableOperation> = Vec::new();
        operations.push(IdentfiableOperation::identity(0));
        operations.push(IdentfiableOperation::hadamard(1));
        operations.push(IdentfiableOperation::phase(2));
        operations.push(IdentfiableOperation::not(3));
        operations.push(IdentfiableOperation::pauli_y(4));
        operations.push(IdentfiableOperation::pauli_z(5));
        operations.push(IdentfiableOperation::cnot(5, 7));
        operations.push(IdentfiableOperation::swap(5, 9));
        display_circuit(operations, 10);
    }
}

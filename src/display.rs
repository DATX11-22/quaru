use crate::operation::{self, Operation, QuantumOperation};

/// Displays a list of operations as a circuit diagram
/// Each horizontal line represents a qubit
/// Each vertical line represents an operation
/// Takes a list of operations and the number of qubits in the circuit
pub fn display_circuit(operations: Vec<IdentfiableOperation>, qubits: usize) {
    let mut circuit: Vec<String> = Vec::new();

    // Push strings to circuit and initialize them with qubit numbers
    // Intersperse each line with an empty line
    for _ in 0..qubits + (qubits - 1) {
        circuit.push(String::new());
    }
    for i in 0..circuit.len() {
        if i % 2 == 1 {
            circuit[i].push_str("    ");
        } else {
            circuit[i].push_str(&format!("|q{}>", i/2));
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
                    circuit[target1 * 2].push_str("X");
                    circuit[target2 * 2].push_str("O"); // Control
                }
                OperationIdentifier::Swap => {
                    // Write Swap symbol
                    circuit[target1 * 2].push_str("X");
                    circuit[target2 * 2].push_str("X");
                }
                _ => {}
            }

            for i in 0..circuit.len() {
                if i == target1*2 || i == target2*2 {
                    continue;
                }
                if (target1*2..target2*2).contains(&i) || (target2*2..target1*2).contains(&i) {
                    circuit[i].push_str("|");
                } else if i % 2 == 0 {
                    circuit[i].push_str("―");
                } else {
                    circuit[i].push_str(" ");
                }
            }
        } else {
            // Unary operation
            let target = idop.operation.targets()[0];
            // Match what operation it is, then write correct symbol
            match idop.identifier {
                OperationIdentifier::Identity => {
                    // Write Identity symbol
                    circuit[target * 2].push_str("I");
                }
                OperationIdentifier::Hadamard => {
                    // Write Hadamard symbol
                    circuit[target * 2].push_str("H");
                }
                OperationIdentifier::Phase => {
                    // Write Phase symbol
                    circuit[target * 2].push_str("P");
                }
                OperationIdentifier::Not => {
                    // Write Not symbol
                    circuit[target * 2].push_str("X");
                }
                OperationIdentifier::PauliY => {
                    // Write Pauli Y symbol
                    circuit[target * 2].push_str("Y");
                }
                OperationIdentifier::PauliZ => {
                    // Write Pauli Z symbol
                    circuit[target * 2].push_str("Z");
                }
                _ => {}
            }
            for i in 0..circuit.len() {
                if i == target * 2 {
                    continue;
                }
                if i % 2 == 0 {
                    circuit[i].push_str("―");
                } else {
                    circuit[i].push_str(" ");
                }
            }
        }
    }
    circuit.iter().for_each(|l| println!("{}", l));
}

/// An operation with an identifier.
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
}

/// Identifier for available operations in the CLI.
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
    /// The CNOT identifier.
    CNot,
    /// The Swap identifier.
    Swap,
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

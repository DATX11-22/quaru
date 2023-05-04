//! The `operation` module provides quantum operation
//! capabilities to a quantum register using matrix representations.
//!
//! # Examples
//! You can explicitly create an [`Operation`](struct@Operation) by
//! providing a complex matrix and targets to [`Operation::new()`]:
//!
//! ```
//! use ndarray::{array, Array2};
//! use quaru::operation::Operation;
//! use quaru::math::real_arr_to_complex;
//!
//! let matrix = real_arr_to_complex(array![[1.0, 0.0], [0.0, 1.0]]);
//! let targets = vec![0];
//!
//! let identity: Option<Operation> = Operation::new(matrix, targets);
//! ```
//!
//! If the custom constructed operation is invalid, [`None`] is returned.
//!
//! You can avoid this for already pre-defined operations:
//!
//! ```
//! use quaru::operation::{Operation, identity};
//!
//! let identity: Operation = identity(0);
//! ```
use crate::math::{c64, int_to_state, real_arr_to_complex};
use ndarray::{array, Array2};
use ndarray::{linalg, Array};
use std::arch::x86_64::_SIDD_MASKED_NEGATIVE_POLARITY;
use std::collections::HashSet;
use std::{f64::consts, vec};

// Naming?
pub trait QuantumOperation {
    fn matrix(&self) -> Array2<c64>;
    fn targets(&self) -> Vec<usize>;
    fn arity(&self) -> usize;
}

/// A quantum computer operation represented by a matrix and targets.
///
/// - `matrix` corresponds to the quantum operator
/// - `targets` corresponds to the operator operands
///
/// # Note
/// In order for an operation to be considered valid, the matrix shape must be square
/// with length equal to the number of operands.
#[derive(Clone, Debug, PartialEq)]
pub struct Operation {
    matrix: Array2<c64>,
    targets: Vec<usize>,
}

impl Operation {
    /// Constructs an operation with the given matrix and targets.
    ///
    /// Returns an operation if `matrix` is square with sides equal to number of `targets`.
    ///
    /// Otherwise, returns `None`.
    pub fn new(matrix: Array2<c64>, targets: Vec<usize>) -> Option<Operation> {
        let shape = matrix.shape();
        let len = targets.len();

        if shape[0] != 2_usize.pow(len as u32) || shape[1] != 2_usize.pow(len as u32) {
            return None;
        }

        Some(Operation { matrix, targets })
    }
}

/// A quantum circuit consisting of a sequence of operations.
/// The circuit can be applied to a quantum register.
/// 
/// - `operations` corresponds to the sequence of operations
/// - `measurement_targets` corresponds to the targets of the measurement operations
/// - `conditional_operations` corresponds to the sequence of conditional operations
/// 
pub struct QuantumCircuit {
    pub(crate) operations: Vec<Operation>,
    measurement_targets: Vec<usize>,
    conditional_operations: Vec<(Operation, Vec<(usize, bool)>)>,
}

impl QuantumCircuit {
    /// Constructs a new empty quantum circuit.
    pub fn new() -> QuantumCircuit {
        QuantumCircuit {
            operations: Vec::new(),
            measurement_targets: Vec::new(),
            conditional_operations: Vec::new(),
        }
    }
    /// Clears the operations in the quantum circuit.
    pub fn reset_register(&mut self) -> &mut Self {
        self.operations.clear();
        self.measurement_targets.clear();
        self.conditional_operations.clear();
        self
    }
    pub fn get_operations(&self) -> &Vec<Operation> {
        &self.operations
    }
    pub fn get_measurement_targets(&mut self) -> &Vec<usize> {
        &self.measurement_targets
    }
    pub fn get_conditional_operations(&mut self) -> &Vec<(Operation, Vec<(usize, bool)>)> {
        &self.conditional_operations
    }
    /// Reduces the circuit by cancelling gates that cancel each other out.
    pub fn reduce_circuit_cancel_gates(&mut self) -> &mut Self {
        let mut i = 0;
        let ops = &self.operations.clone();
        let mut new_ops: Vec<Operation> = Vec::new();
        while i < ops.len() {
            if i == ops.len() - 1 {
                new_ops.push(ops[i].clone());
                break;
            }
            let j = i + 1;
            let op = &ops[i];
            let next_op = &ops[j];
            //all real and symetric unitary gates cancel each other out
            if op.matrix().iter().all(|x| x.im == 0.0) && *op == *next_op {
                i += 1;
            } else {
                new_ops.push(op.clone());
            }
            i += 1;
        }
        self.operations = new_ops;
        //If any gates were removed we need to check again
        if self.operations.len() == ops.len() || self.operations.len() <= 1 {
            return self;
        }
        self.reduce_circuit_cancel_gates();
        self
    }
    /// Reduces the circuit by multiplying gates with the same targets.
    pub fn reduce_circuit_gates_with_same_targets(&mut self) -> &mut Self {
        let mut i = 0;
        let ops = &self.operations;
        let mut new_ops: Vec<Operation> = Vec::new();
        let len = ops.len();
        while i < len {
            let op = &ops[i];

            let mut matrix = op.matrix();
            let targets = op.targets().clone();
            let mut j = i + 1;

            while j < self.get_operations().len() && self.get_operations()[j].targets() == targets {
                let next_op = &self.get_operations()[j];
                matrix = next_op.matrix().dot(&matrix);
                j += 1;
            }
            new_ops.push(Operation::new(matrix, targets).expect("Could not create operation"));
            i = j;
        }
        self.operations = new_ops;
        self
    }

    /// Reduces the circuit by combining gates that are not overlapping, essentially putting them in the same time slot without altering the program.
    pub fn reduce_non_overlapping_gates(&mut self) -> &mut Self {
        let mut i = 0;
        let ops = &self.operations;
        let mut new_ops: Vec<Operation> = Vec::new();
        let len = ops.len();
        while i < len {
            let op = &ops[i];
            let mut matrix = op.matrix();
            let mut targets: Vec<usize> = Vec::new();
            targets.append(&mut op.targets().clone());
            let mut j = i + 1;
            while j < len && !targets.iter().any(|x| ops[j].targets().contains(x)) {
                let next_op = &self.get_operations()[j];
                matrix = linalg::kron(&next_op.matrix(), &matrix).to_owned();
                targets.append(&mut next_op.targets().clone());

                j += 1;
            }
            new_ops.push(
                Operation::new(matrix.clone(), targets.to_vec())
                    .expect("Could not create operation"),
            );
            i = j;
        }
        self.operations = new_ops;
        self
    }
    
    pub fn clear_operations(&mut self) {
        self.operations.clear();
    }
    /// Adds an operation to the circuit.
    /// ***Panics*** if the operation is added after a measurement.
    pub fn add_operation(&mut self, operation: Operation) {
        if operation
            .targets()
            .iter()
            .any(|x| self.measurement_targets.contains(x))
        {
            panic!("Cannot add operation after measurement");
        }
        if self.conditional_operations.len() > 0 {
            panic!("Cannot add operation after conditional operation (yet...)");
        }
        self.operations.push(operation);
    }

    /// Adds a measurement to the circuit.
    /// ***Panics*** if the measurement is added after another measurement.
    pub fn add_measurement(&mut self, target: usize) {
        if self.measurement_targets.contains(&target) {
            panic!("Cannot add measurement after measurement");
        }
        self.measurement_targets.push(target);
    }

    /// Adds a conditional operation to the circuit.
    /// 
    /// ***Panics*** if the operation is with a target that is not already measured.
    pub fn add_conditional_operation(
        &mut self,
        operation: Operation,
        condition: Vec<(usize, bool)>,
    ) {
        if operation
            .targets()
            .iter()
            .any(|x| self.measurement_targets.contains(x))
        {
            panic!("Cannot add operation after measurement");
        }
        if !condition
            .iter()
            .all(|x: &(usize, bool)| self.measurement_targets.contains(&x.0))
        {
            panic!("Condition must be a measurement");
        }
        self.conditional_operations.push((operation, condition));
    }
}

// TODO: Check if we can return references instead?
impl QuantumOperation for Operation {
    fn matrix(&self) -> Array2<c64> {
        self.matrix.clone()
    }

    fn targets(&self) -> Vec<usize> {
        self.targets.clone()
    }

    fn arity(&self) -> usize {
        self.targets().len()
    }
}

/// Returns the identity operation for some `target` qubit.
pub fn identity(target: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(array![[1.0, 0.0], [0.0, 1.0]]),
        targets: vec![target],
    }
}

/// Returns the hadamard operation for the given `target` qubit.
///
/// Creates an equal superposition of the target qubit's basis states.
pub fn hadamard(target: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(consts::FRAC_1_SQRT_2 * array![[1.0, 1.0], [1.0, -1.0]]),
        targets: vec![target],
    }
}

/// Returns a hadamard transformation for the given qubit `targets`.
pub fn hadamard_transform(targets: Vec<usize>) -> Operation {
    let mut matrix = hadamard(targets[0]).matrix();
    let len = targets.len();

    for t in targets.iter().take(len).skip(1) {
        matrix = linalg::kron(&hadamard(*t).matrix(), &matrix);
    }
    Operation { matrix, targets }
}

/// Returns the controlled NOT operation based on the given `control` qubit and
/// `target` qubit.
///
/// Flips the target qubit if and only if the control qubit is |1⟩.
pub fn cnot(control: usize, target: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ]),
        targets: vec![target, control],
    }
}

/// Create a quantum gate from a function.
/// The c:th column of the matrix contains f(c) in binary.
pub fn to_quantum_gate(f: &dyn Fn(usize) -> usize, targets: Vec<usize>) -> Operation {
    let t_len = targets.len();
    let len: usize = 1 << t_len;
    let mut matrix: Array2<c64> = Array2::zeros((len, len));
    // Loop through the columns
    for c in 0..len {
        let val = f(c);
        let res_state = int_to_state(val, len);

        // Set each cell in the column
        for r in 0..len {
            matrix[(r, c)] = res_state[(r, 0)];
        }
    }
    Operation { matrix, targets }
}

/// Create a controlled version of an operation.
/// Doubles the width and height of the matrix, put the original matrix
/// in the bottom right corner and add an identity matrix in the top left corner.
pub fn to_controlled(op: Operation, control: usize) -> Operation {
    let old_sz = 1 << op.arity();
    let mut matrix = Array2::zeros((2 * old_sz, 2 * old_sz));
    for i in 0..old_sz {
        matrix[(i, i)] = c64::new(1.0, 0.0);
    }
    for i in 0..old_sz {
        for j in 0..old_sz {
            matrix[(i + old_sz, j + old_sz)] = op.matrix[(i, j)];
        }
    }
    let mut targets = op.targets();

    // One more target bit: the control.
    targets.push(control);
    Operation { matrix, targets }
}

/// Returns the swap operation for the given target qubits.
///
/// Swaps two qubits in the register.
pub fn swap(target_1: usize, target_2: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        targets: vec![target_1, target_2],
    }
}

/// Returns the phase operation for the given `target` qubit.
///
/// Maps the basis states |0⟩ -> |0⟩ and |1⟩ -> i|1⟩, modifying the
/// phase of the quantum state.
pub fn phase(target: usize) -> Operation {
    Operation {
        matrix: array![
            [c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
            [c64::new(0.0, 0.0), c64::new(0.0, 1.0)]
        ],
        targets: vec![target],
    }
}

/// Returns the NOT operation for the given `target` qubit.
///
/// Maps the basis states |0⟩ -> |1⟩ and |1⟩ -> |0⟩.
///
/// Also referred to as the Pauli-X operation.
pub fn not(target: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(array![[0.0, 1.0], [1.0, 0.0]]),
        targets: vec![target],
    }
}

/// Returns the Pauli-Y operation for a given `target` qubit.
///
/// Maps the basis states |0⟩ -> i|1⟩ and |1⟩ -> -i|0⟩.
pub fn pauli_y(target: usize) -> Operation {
    Operation {
        matrix: array![
            [c64::new(0.0, 0.0), c64::new(0.0, -1.0)],
            [c64::new(0.0, 1.0), c64::new(0.0, 0.0)]
        ],
        targets: vec![target],
    }
}

/// Returns the Pauli-Z operation for a given `target` qubit.
///
/// Maps the basis states |0⟩ -> |0⟩ and |1⟩ -> -|1⟩
pub fn pauli_z(target: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(array![[1.0, 0.0], [0.0, -1.0]]),
        targets: vec![target],
    }
}

/// Returns the controlled NOT operation for the given number of `control` qubits on the `target` qubit.
///
/// Flips the target qubit if and only if controls are |1⟩.
pub fn cnx(controls: &[usize], target: usize) -> Operation {
    let mut targets = vec![target];
    targets.append(&mut controls.to_owned());

    // Calculates the size of the matrix (2^n) where n is the number of target + control qubits
    let n: usize = 2_usize.pow(targets.len() as u32);

    // Creates an empty (2^n * 2^n) matrix and starts to fill it in as an identity matrix
    let mut matrix: Array2<f64> = Array2::<f64>::zeros((n, n));
    for i in 0..n - 2 {
        // Does not fill in the last two rows
        matrix.row_mut(i)[i] = 1.0;
    }

    // The last two rows are to be "swapped", finalizing the not part of the matrix
    matrix.row_mut(n - 1)[n - 2] = 1.0;
    matrix.row_mut(n - 2)[n - 1] = 1.0;

    Operation {
        matrix: real_arr_to_complex(matrix),
        targets,
    }
}

/// Returns a controlled Pauli-Z operation for the given number of `control` qubits on the `target`
/// qubit.
///
/// Maps the basis states of the target to |0⟩ -> |0⟩ and |1⟩ -> -|1⟩ if and only if the controls
/// are |1⟩.
pub fn cnz(controls: &[usize], target: usize) -> Operation {
    let mut targets = vec![target];
    targets.append(&mut controls.to_owned());

    let n: usize = 2_usize.pow(targets.len() as u32);

    let mut matrix: Array2<f64> = Array2::<f64>::zeros((n, n));
    for i in 0..n - 1 {
        matrix.row_mut(i)[i] = 1.0;
    }
    matrix.row_mut(n - 1)[n - 1] = -1.0;

    Operation {
        matrix: real_arr_to_complex(matrix),
        targets,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cnot, cnx, hadamard, identity, not, pauli_y, pauli_z, phase, swap, QuantumCircuit,
        QuantumOperation,
    };
    use crate::math::c64;
    use ndarray::Array2;

    fn all_ops() -> Vec<Box<dyn QuantumOperation>> {
        vec![
            Box::new(identity(0)),
            Box::new(hadamard(0)),
            Box::new(cnot(0, 1)),
            Box::new(swap(0, 1)),
            Box::new(phase(0)),
            Box::new(not(0)),
            Box::new(pauli_y(0)),
            Box::new(pauli_z(0)),
            Box::new(cnx(&[0], 1)),
            Box::new(cnx(&[0, 1], 2)),
            Box::new(cnx(&[0, 1, 2], 3)),
            Box::new(cnx(&[0, 1, 2, 3], 4)),
            Box::new(cnx(&[0, 1, 2, 3, 4], 5)),
        ]
    }

    #[test]
    fn sz_matches() {
        for op in all_ops() {
            assert_eq!(op.matrix().dim().0, op.matrix().dim().1);
            assert_eq!(op.matrix().dim().0, 1 << op.arity())
        }
    }

    #[test]
    fn unitary() {
        // This also guarantees preservation of total probability
        for op in all_ops() {
            let conj_transpose = op.matrix().t().map(|e| e.conj());
            assert!(matrix_is_equal(
                op.matrix().dot(&conj_transpose),
                Array2::eye(op.matrix().dim().0),
                1e-8
            ))
        }
    }

    #[test]
    fn toffoli2_equals_cnot() {
        let toffoli_generated_cnot = cnx(&[0], 1);
        assert!(matrix_is_equal(
            toffoli_generated_cnot.matrix(),
            cnot(0, 1).matrix(),
            1e-8
        ));
    }

    fn matrix_is_equal(a: Array2<c64>, b: Array2<c64>, tolerance: f64) -> bool {
        (a - b).iter().all(|e| e.norm() < tolerance)
    }

    #[test]
    fn test_two_pair_of_same_gates_cancel_with_one_pair_in_middle() {
        let mut circuit = QuantumCircuit::new();

        circuit.add_operation(hadamard(0));

        circuit.add_operation(cnot(0, 1));
        circuit.add_operation(cnot(0, 1));

        circuit.add_operation(hadamard(0));

        circuit.reduce_circuit_cancel_gates();
        assert!(circuit.get_operations().len() == 0);
    }
    #[test]
    fn test_gates_with_same_arity_and_target_mul_together() {
        let mut circuit = QuantumCircuit::new();

        //Will multiply together
        circuit.add_operation(hadamard(0));
        circuit.add_operation(not(0));

        circuit.reduce_circuit_gates_with_same_targets();

        assert!(circuit.get_operations().len() == 1);
    }

    #[test]
    fn test_hadamard_twice_cancels() {
        let mut circuit = QuantumCircuit::new();

        //Will cancel
        circuit.add_operation(hadamard(0));
        circuit.add_operation(hadamard(0));

        circuit.reduce_circuit_cancel_gates();

        assert!(circuit.get_operations().len() == 0);
    }

    #[test]
    fn test_cnot_twice_cancels() {
        let mut circuit = QuantumCircuit::new();

        //Will cancel
        circuit.add_operation(cnot(0, 1));
        circuit.add_operation(cnot(0, 1));

        circuit.reduce_circuit_cancel_gates();

        assert!(circuit.get_operations().len() == 0);
    }
}

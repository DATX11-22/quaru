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
//! use quaru::math::real_to_complex;
//!
//! let matrix = real_to_complex(array![[1.0, 0.0], [0.0, 1.0]]);
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
use ndarray::{array, Array2};
use std::{f64::consts, vec};
use crate::math::{real_arr_to_complex, c64, new_complex};

// Naming?
pub trait OperationTrait {
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
#[derive(Clone, Debug)]
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

// TODO: Check if we can return references instead?
impl OperationTrait for Operation {
    fn matrix(&self) -> Array2<c64> {
        self.matrix.clone()
    }

    fn targets(&self) -> Vec<usize> {
        self.targets.to_vec()
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
            [new_complex(1.0, 0.0), new_complex(0.0, 0.0)],
            [new_complex(0.0, 0.0), new_complex(0.0, 1.0)]
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
            [new_complex(0.0, 0.0), new_complex(0.0, -1.0)],
            [new_complex(0.0, 1.0), new_complex(0.0, 0.0)]
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

pub fn u(theta: f64, phi: f64, lambda: f64, target: usize) -> Operation {
    let theta = c64::from(theta);
    let phi = c64::from(phi);
    let lambda = c64::from(lambda);
    let i = c64::i();
    Operation {
        matrix: array![
            [(-i * (phi+lambda) / 2.0).exp() * (theta / 2.0).cos(), -(-i * (phi-lambda) / 2.0).exp() * (theta / 2.0).sin()],
            [( i * (phi-lambda) / 2.0).exp() * (theta / 2.0).sin(),  ( i * (phi+lambda) / 2.0).exp() * (theta / 2.0).cos()],
        ],
        targets: vec![target],
    }
}

#[cfg(test)]
mod tests {
    use super::{cnx, OperationTrait};
    use ndarray::Array2;
    use crate::math::c64;

    use super::{cnot, hadamard, identity, not, pauli_y, pauli_z, phase, swap};

    fn all_ops() -> Vec<Box<dyn OperationTrait>> {
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
}

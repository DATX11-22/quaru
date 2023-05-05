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
use crate::math::{c64, int_to_state, real_arr_to_complex, ndarray_to_arrayfire, arrayfire_to_ndarray};
use ndarray::linalg;
use ndarray::{array, Array2};
use num::Zero;
use std::{f64::consts, vec};
extern crate arrayfire as af;

// Naming?
pub trait QuantumOperation {
    fn matrix(&self) -> af::Array<c64>;
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
    matrix: af::Array<c64>,
    targets: Vec<usize>,
}

impl Operation {
    /// Constructs an operation with the given matrix and targets.
    ///
    /// Returns an operation if `matrix` is square with sides equal to number of `targets`.
    ///
    /// Otherwise, returns `None`.
    pub fn new(matrix: af::Array<c64>, targets: Vec<usize>) -> Option<Operation> {
        let shape = matrix.dims();
        let len = targets.len();

        if shape[0] as usize != 2_usize.pow(len as u32) || shape[1] as usize != 2_usize.pow(len as u32) {
            return None;
        }

        Some(Operation { matrix, targets })
    }
}

// TODO: Check if we can return references instead?
impl QuantumOperation for Operation {
    fn matrix(&self) -> af::Array<c64> {
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

/// Returns a hadamard transformation for the given qubit `targets`.
pub fn hadamard_transform(targets: Vec<usize>) -> Operation {
    let mut matrix = array![[c64::new(1.0, 0.0)]];
    let len = targets.len();

    // TODO: faster to construct on GPU?
    for t in targets.iter().take(len) {
        matrix = linalg::kron(&arrayfire_to_ndarray(&hadamard(*t).matrix()), &matrix);
    }
    let af_matrix = ndarray_to_arrayfire(&matrix);
    Operation { matrix: af_matrix, targets }
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
    // TODO: faster to construct on GPU?
    for c in 0..len {
        let val = f(c);
        let res_state = int_to_state(val, len);

        // Set each cell in the column
        for r in 0..len {
            matrix[(r, c)] = res_state[(r, 0)];
        }
    }
    let af_matrix = ndarray_to_arrayfire(&matrix);
    Operation { matrix: af_matrix, targets }
}

/// Create a controlled version of an operation.
/// Doubles the width and height of the matrix, put the original matrix
/// in the bottom right corner and add an identity matrix in the top left corner.
pub fn to_controlled(op: Operation, control: usize) -> Operation {
    // TODO: Classic optization?
    /*
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
    */

    let old_sz = 1 << op.arity();
    let mut matrix = af::constant!(c64::zero(); 2 * old_sz, 2 * old_sz);
    let identity = af::identity(af::dim4!(old_sz, old_sz));

    let row_seq = af::Seq::new(0, old_sz as u32 - 1, 1);
    let col_seq = af::Seq::new(0, old_sz as u32 - 1, 1);
    af::assign_seq(&mut matrix, &[row_seq, col_seq], &identity);

    let row_seq = af::Seq::new(old_sz as u32, 2 * old_sz as u32 - 1, 1);
    let col_seq = af::Seq::new(old_sz as u32, 2 * old_sz as u32 - 1, 1);
    af::assign_seq(&mut matrix, &[row_seq, col_seq], &op.matrix());

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
        matrix: af::Array::new(
            &[c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 1.0)],
            af::dim4!(2, 2)
        ),
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
        matrix: af::Array::new(
            &[c64::new(0.0, 0.0), c64::new(0.0, -1.0), c64::new(0.0, 1.0), c64::new(0.0, 0.0)],
            af::dim4!(2, 2)
        ),
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
        cnot, cnx, hadamard, identity, not, pauli_y, pauli_z, phase, swap, QuantumOperation,
    };
    use crate::math::{c64, arrayfire_to_ndarray};
    use af::MatProp;
    use ndarray::Array2;
    extern crate arrayfire as af;

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
            assert_eq!(op.matrix().dims()[0], op.matrix().dims()[1]);
            assert_eq!(op.matrix().dims()[0], 1 << op.arity())
        }
    }

    #[test]
    fn unitary() {
        // This also guarantees preservation of total probability
        for op in all_ops() {
            let conj_transpose = af::transpose(&op.matrix(), true);
            assert!(matrix_is_equal(
                arrayfire_to_ndarray(&af::matmul(&op.matrix(), &conj_transpose, MatProp::NONE, MatProp::NONE)),
                Array2::eye(op.matrix().dims()[0] as usize),
                1e-8
            ))
        }
    }

    #[test]
    fn toffoli2_equals_cnot() {
        let toffoli_generated_cnot = cnx(&[0], 1);
        assert!(matrix_is_equal(
            arrayfire_to_ndarray(&toffoli_generated_cnot.matrix()),
            arrayfire_to_ndarray(&cnot(0, 1).matrix()),
            1e-8
        ));
    }

    fn matrix_is_equal(a: Array2<c64>, b: Array2<c64>, tolerance: f64) -> bool {
        (a - b).iter().all(|e| e.norm() < tolerance)
    }
}

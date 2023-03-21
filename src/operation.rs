use ndarray::{array, Array2};
use num::Complex;
use std::{f64::consts, vec};

use crate::math::real_to_complex;

// Naming?
pub trait OperationTrait {
    fn matrix(&self) -> Array2<Complex<f64>>;
    fn targets(&self) -> Vec<usize>;
    fn arity(&self) -> usize;
}

/// A quantum computer operation represented by a matrix and targets.
///
/// - `matrix` corresponds to the quantum operator
/// - `targets` corresponds to the operator operands
#[derive(Clone, Debug)]
pub struct Operation {
    matrix: Array2<Complex<f64>>,
    targets: Vec<usize>,
}

impl Operation {
    /// Constructs an operation with the given matrix, targets and arity.
    ///
    /// Returns an operation if:
    /// - `matrix` is square with sides equal to number of `targets`
    /// - number of `targets` is equal to `arity`
    ///
    /// Otherwise, returns `None`.
    pub fn new(
        matrix: Array2<Complex<f64>>,
        targets: Vec<usize>,
        arity: usize,
    ) -> Option<Operation> {
        if targets.len() != arity {
            return None;
        }

        let shape = matrix.shape();
        let len = targets.len();

        if shape[0] != len || shape[1] != len {
            return None;
        }

        Some(Operation {
            matrix,
            targets,
        })
    }
}

// TODO: Check if we can return references instead?
impl OperationTrait for Operation {
    fn matrix(&self) -> Array2<Complex<f64>> {
        self.matrix.clone()
    }

    fn targets(&self) -> Vec<usize> {
        self.targets.to_vec()
    }

    fn arity(&self) -> usize {
        self.targets().len()
    }
}

pub fn identity(target: usize) -> Operation {
    Operation {
        matrix: real_to_complex(array![[1.0, 0.0], [0.0, 1.0]]),
        targets: vec![target],
    }
}

pub fn hadamard(target: usize) -> Operation {
    Operation {
        matrix: real_to_complex(consts::FRAC_1_SQRT_2 * array![[1.0, 1.0], [1.0, -1.0]]),
        targets: vec![target],
    }
}

pub fn cnot(control: usize, target: usize) -> Operation {
    Operation {
        matrix: real_to_complex(array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ]),
        targets: vec![target, control],
    }
}

pub fn swap(target1: usize, target2: usize) -> Operation {
    Operation {
        matrix: real_to_complex(array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        targets: vec![target1, target2],
    }
}

pub fn phase(target: usize) -> Operation {
    Operation {
        matrix: array![
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)]
        ],
        targets: vec![target],
    }
}

pub fn not(target: usize) -> Operation {
    Operation {
        matrix: real_to_complex(array![[0.0, 1.0], [1.0, 0.0]]),
        targets: vec![target],
    }
}

pub fn pauli_y(target: usize) -> Operation {
    Operation {
        matrix: array![
            [Complex::new(0.0, 0.0), Complex::new(0.0, -1.0)],
            [Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)]
        ],
        targets: vec![target],
    }
}

pub fn pauli_z(target: usize) -> Operation {
    Operation {
        matrix: real_to_complex(array![[1.0, 0.0], [0.0, -1.0]]),
        targets: vec![target],
    }
}

pub fn toffoli(controls: &Vec<usize>, target: usize) -> Operation {
    let mut targets = vec![target];
    targets.append(&mut controls.clone());

    // Calculates the size of the matrix (2^n) where n is the number of target + control qubits
    let n: usize = (2 as usize).pow(targets.len() as u32);

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
        matrix: real_to_complex(matrix),
        targets,
    }
}

pub fn oracle_operation(regsize: usize, winner: usize) -> Operation {
    let n: usize = (2 as usize).pow(regsize as u32);
    let mut matrix: Array2<f64> = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        matrix.row_mut(i)[i] = if i == winner { -1.0 } else { 1.0 };
    }

    Operation {
        matrix: real_to_complex(matrix),
        targets: vec![0],
    }
}

pub fn cz(controls: &Vec<usize>, target: usize) -> Operation {
    let mut targets = vec![target];
    targets.append(&mut controls.clone());

    let n: usize = (2 as usize).pow(targets.len() as u32);

    let mut matrix: Array2<f64> = Array2::<f64>::zeros((n, n));
    for i in 0..n - 1 {
        matrix.row_mut(i)[i] = 1.0;
    }
    matrix.row_mut(n - 1)[n - 1] = -1.0;

    Operation {
        matrix: real_to_complex(matrix),
        targets,
    }
}


#[cfg(test)]
mod tests {
    use super::{toffoli, OperationTrait};
    use ndarray::Array2;
    use num::Complex;

    use super::{cnot, hadamard, identity, not, pauli_y, pauli_z, phase, swap};

    fn all_ops() -> Vec<Box<dyn OperationTrait>> {
        return vec![
            Box::new(identity(0)),
            Box::new(hadamard(0)),
            Box::new(cnot(0, 1)),
            Box::new(swap(0, 1)),
            Box::new(phase(0)),
            Box::new(not(0)),
            Box::new(pauli_y(0)),
            Box::new(pauli_z(0)),
            Box::new(toffoli(&vec![0], 1)),
            Box::new(toffoli(&vec![0, 1], 2)),
            Box::new(toffoli(&vec![0, 1, 2], 3)),
            Box::new(toffoli(&vec![0, 1, 2, 3], 4)),
            Box::new(toffoli(&vec![0, 1, 2, 3, 4], 5)),
        ];
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
        let toffoli_generated_cnot = toffoli(&vec![0], 1);
        assert!(matrix_is_equal(
            toffoli_generated_cnot.matrix(),
            cnot(0, 1).matrix(),
            1e-8
        ));
    }

    fn matrix_is_equal(a: Array2<Complex<f64>>, b: Array2<Complex<f64>>, tolerance: f64) -> bool {
        (a - b).iter().all(|e| e.norm() < tolerance)
    }
}

use ndarray::{array, Array2};
use num::Complex;
use std::{f64::consts, vec};

// Naming?
pub trait OperationTrait {
    fn matrix(&self) -> Array2<Complex<f64>>;
    fn targets(&self) -> Vec<usize>;
    fn arity(&self) -> usize;
}

pub struct Operation<const ARITY: usize> {
    matrix: Array2<Complex<f64>>,
    targets: [usize; ARITY],
}

// TODO: Check if we can return references instead?
impl<const ARITY: usize> OperationTrait for Operation<ARITY> {
    fn matrix(&self) -> Array2<Complex<f64>> {
        self.matrix.clone()
    }

    fn targets(&self) -> Vec<usize> {
        self.targets.to_vec()
    }

    fn arity(&self) -> usize {
        ARITY
    }
}

pub fn identity(target: usize) -> Operation<1> {
    Operation {
        matrix: real_to_complex(array![[1.0, 0.0], [0.0, 1.0]]),
        targets: [target],
    }
}

pub fn hadamard(target: usize) -> Operation<1> {
    Operation {
        matrix: real_to_complex(consts::FRAC_1_SQRT_2 * array![[1.0, 1.0], [1.0, -1.0]]),
        targets: [target],
    }
}

pub fn cnot(control: usize, target: usize) -> Operation<2> {
    Operation {
        matrix: real_to_complex(array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ]),
        targets: [control, target],
    }
}

pub fn swap(target1: usize, target2: usize) -> Operation<2> {
    Operation {
        matrix: real_to_complex(array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        targets: [target1, target2],
    }
}

pub fn phase(target: usize) -> Operation<1> {
    Operation {
        matrix: array![
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)]
        ],
        targets: [target],
    }
}

pub fn not(target: usize) -> Operation<1> {
    Operation {
        matrix: real_to_complex(array![[0.0, 1.0], [1.0, 0.0]]),
        targets: [target],
    }
}

pub fn pauli_y(target: usize) -> Operation<1> {
    Operation {
        matrix: array![
            [Complex::new(0.0, 0.0), Complex::new(0.0, -1.0)],
            [Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)]
        ],
        targets: [target],
    }
}

pub fn pauli_z(target: usize) -> Operation<1> {
    Operation {
        matrix: real_to_complex(array![[1.0, 0.0], [0.0, -1.0]]),
        targets: [target],
    }
}

fn real_to_complex(matrix: Array2<f64>) -> Array2<Complex<f64>> {
    matrix.map(|e| e.into())
}

#[cfg(test)]
mod tests {
    use super::OperationTrait;
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

    fn matrix_is_equal(a: Array2<Complex<f64>>, b: Array2<Complex<f64>>, tolerance: f64) -> bool {
        (a - b).iter().all(|e| e.norm() < tolerance)
    }
}

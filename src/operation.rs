use ndarray::{array, Array2};
use num::Complex;
use std::f64::consts;

pub struct Operation<const ARITY: usize> {
    matrix: Array2<Complex<f64>>,
    targets: [usize; ARITY],
}

// TODO: Check if we can return references instead?
impl<const ARITY: usize> Operation<ARITY> {
    pub fn matrix(&self) -> Array2<Complex<f64>> {
        self.matrix.clone()
    }

    pub fn targets(&self) -> [usize; ARITY] {
        self.targets.clone()
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

fn real_to_complex(matrix: Array2<f64>) -> Array2<Complex<f64>> {
    matrix.map(|e| e.into())
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num::Complex;

    use super::{cnot, hadamard, identity, not, phase, swap};

    fn all_ops() -> Vec<(Array2<Complex<f64>>, usize)> {
        return vec![
            (identity(0).matrix(), identity(0).targets.len()),
            (hadamard(0).matrix(), hadamard(0).targets.len()),
            (cnot(0, 1).matrix(), cnot(0, 1).targets.len()),
            (swap(0, 1).matrix(), swap(0, 1).targets.len()),
            (phase(0).matrix(), phase(0).targets.len()),
            (not(0).matrix(), not(0).targets.len()),
        ];
    }

    #[test]
    fn sz_matches() {
        for (matrix, arity) in all_ops() {
            assert_eq!(matrix.dim().0, matrix.dim().1);
            assert_eq!(matrix.dim().0, 1 << arity)
        }
    }

    #[test]
    fn unitary() {
        // This also guarantees preservation of total probability
        for (mat, _) in all_ops() {
            let conj_transpose = mat.t().map(|e| e.conj());
            assert!(matrix_is_equal(
                mat.dot(&conj_transpose),
                Array2::eye(mat.dim().0),
                1e-8
            ))
        }
    }

    fn matrix_is_equal(a: Array2<Complex<f64>>, b: Array2<Complex<f64>>, tolerance: f64) -> bool {
        (a - b).iter().all(|e| e.norm() < tolerance)
    }
}

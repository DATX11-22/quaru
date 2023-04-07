use ndarray::{array, Array2};
use std::{f64::consts, vec};
use crate::math::{real_arr_to_complex, c64, new_complex};

use num::Complex;
use num::complex::ComplexFloat;
use ndarray::Axis;
use ndarray::linalg;

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

pub fn identity(target: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(array![[1.0, 0.0], [0.0, 1.0]]),
        targets: vec![target],
    }
}

pub fn hadamard(target: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(consts::FRAC_1_SQRT_2 * array![[1.0, 1.0], [1.0, -1.0]]),
        targets: vec![target],
    }
}

pub fn hadamard_transform(targets: Vec<usize>) -> Operation {
    let mut matrix = hadamard(targets[0]).matrix();
    let len = targets.len();

    for i in 1..len {
        matrix = linalg::kron(&hadamard(targets[i]).matrix(), &matrix);
    }
    Operation {
        matrix: matrix,
        targets: targets,
    }
}

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

pub fn qft(n: usize) -> Operation {
    let m = 1 << n;
    let mut matrix = Array2::zeros((m, m));
    let w = consts::E.powc(Complex::new(0.0, 2.0 * consts::PI / m as f64));
    for i in 0..m as i32 {
        for j in 0..m as i32 {
            matrix[(i as usize, j as usize)] = w.powi(i * j) * (1.0 / (m as f64).sqrt());
        }
    }
    Operation {
        matrix: matrix,
        targets: (0..n).collect(),
    }
}

pub fn to_quantum_gate(f: &dyn Fn(i32) -> i32, targets: Vec<usize>) -> Operation {
    let t_len = targets.len();
    let len: usize = 1 << t_len;
    let mut matrix: Array2<Complex<f64>> = Array2::zeros((len, len));
    for i in 0..len {
        let val = f(i as i32);
        let res_state = to_state(val as u32, len);
        for j in 0..len {
            let res_val = res_state[(j, 0)];
            matrix[(j, i)] = res_val;
        }
    }
    Operation {
        matrix: matrix,
        targets: targets,
    }
}

fn to_state(val: u32, len: usize) -> Array2<Complex<f64>> {
    let mut state: Array2<Complex<f64>> = Array2::zeros((len, 1));
    let i = (val % (len as u32)) as usize;
    state[(i, 0)] = Complex::new(1.0, 0.0);

    // println!("from val {} \n{:}", val, state);
    state
}

pub fn to_controlled(op: Operation, control: usize) -> Operation {
    let extend = 2_i32.pow(op.arity() as u32) as usize;
    let mut matrix = Array2::zeros((
        op.matrix().len_of(Axis(0)) + extend,
        op.matrix().len_of(Axis(1)) + extend,
    ));
    for i in 0..extend {
        matrix[(i, i)] = Complex::new(1.0, 0.0);
    }
    for i in 0..op.matrix.len_of(Axis(0)) {
        for j in 0..op.matrix.len_of(Axis(1)) {
            matrix[(i + extend, j + extend)] = op.matrix[(i, j)];
        }
    }
    let mut targets = op.targets();
    targets.push(control);
    Operation {
        matrix: matrix,
        targets: targets,
    }
}

pub fn swap(target1: usize, target2: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(array![
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
            [new_complex(1.0, 0.0), new_complex(0.0, 0.0)],
            [new_complex(0.0, 0.0), new_complex(0.0, 1.0)]
        ],
        targets: vec![target],
    }
}

pub fn not(target: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(array![[0.0, 1.0], [1.0, 0.0]]),
        targets: vec![target],
    }
}

pub fn pauli_y(target: usize) -> Operation {
    Operation {
        matrix: array![
            [new_complex(0.0, 0.0), new_complex(0.0, -1.0)],
            [new_complex(0.0, 1.0), new_complex(0.0, 0.0)]
        ],
        targets: vec![target],
    }
}

pub fn pauli_z(target: usize) -> Operation {
    Operation {
        matrix: real_arr_to_complex(array![[1.0, 0.0], [0.0, -1.0]]),
        targets: vec![target],
    }
}

pub fn toffoli(controls: &Vec<usize>, target: usize) -> Operation {
    let mut targets = vec![target];
    targets.append(&mut controls.clone());

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

pub fn cz(controls: &Vec<usize>, target: usize) -> Operation {
    let mut targets = vec![target];
    targets.append(&mut controls.clone());

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
    use super::{toffoli, OperationTrait};
    use ndarray::Array2;
    use crate::math::c64;

    use super::{cnot, hadamard, identity, not, pauli_y, pauli_z, phase, qft, swap};

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
            Box::new(qft(5)),
            Box::new(toffoli(&vec![0], 1)),
            Box::new(toffoli(&vec![0, 1], 2)),
            Box::new(toffoli(&vec![0, 1, 2], 3)),
            Box::new(toffoli(&vec![0, 1, 2, 3], 4)),
            Box::new(toffoli(&vec![0, 1, 2, 3, 4], 5)),
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
        let toffoli_generated_cnot = toffoli(&vec![0], 1);
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

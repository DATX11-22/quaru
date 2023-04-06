use ndarray::{array, linalg, Array2, Axis};

use num::{complex::ComplexFloat, Complex};
use std::{
    f64::consts::{self, PI},
    vec,
};

// Naming?
pub trait OperationTrait {
    fn matrix(&self) -> Array2<Complex<f64>>;
    fn targets(&self) -> Vec<usize>;
    fn arity(&self) -> usize;
}

#[derive(Clone, Debug)]
pub struct Operation {
    pub matrix: Array2<Complex<f64>>,
    targets: Vec<usize>,
    arity: usize,
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
        self.arity
    }
}

pub fn identity(target: usize) -> Operation {
    Operation {
        matrix: real_to_complex(array![[1.0, 0.0], [0.0, 1.0]]),
        targets: vec![target],
        arity: 1,
    }
}

pub fn hadamard(target: usize) -> Operation {
    Operation {
        matrix: real_to_complex(consts::FRAC_1_SQRT_2 * array![[1.0, 1.0], [1.0, -1.0]]),
        targets: vec![target],
        arity: 1,
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
        arity: len,
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
        // targets: vec![control, target],
        targets: vec![target, control],
        arity: 2,
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
        arity: n,
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
        arity: t_len,
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
    let extend = 2_i32.pow(op.arity as u32) as usize;
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
        arity: op.arity + 1,
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
        arity: 2,
    }
}

pub fn phase(target: usize) -> Operation {
    Operation {
        matrix: array![
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)]
        ],
        targets: vec![target],
        arity: 1,
    }
}

pub fn not(target: usize) -> Operation {
    Operation {
        matrix: real_to_complex(array![[0.0, 1.0], [1.0, 0.0]]),
        targets: vec![target],
        arity: 1,
    }
}

pub fn pauli_y(target: usize) -> Operation {
    Operation {
        matrix: array![
            [Complex::new(0.0, 0.0), Complex::new(0.0, -1.0)],
            [Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)]
        ],
        targets: vec![target],
        arity: 1,
    }
}

pub fn pauli_z(target: usize) -> Operation {
    Operation {
        matrix: real_to_complex(array![[1.0, 0.0], [0.0, -1.0]]),
        targets: vec![target],
        arity: 1,
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

    use super::{cnot, hadamard, identity, not, pauli_y, pauli_z, phase, qft, swap};

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
            Box::new(qft(5)),
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

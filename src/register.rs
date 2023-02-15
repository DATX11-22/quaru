use crate::{
    math,
    operation::{self, Operation},
};
use ndarray::{array, linalg, Array2};
use num::Complex;
use rand::prelude::*;

#[derive(Clone, Debug)]
pub struct Register<const N: usize> {
    pub state: Array2<Complex<f64>>, // Should not be pub (it is pub now for testing purpouses)
}

impl<const N: usize> Register<N> {
    pub fn new(input_bits: [bool; N]) -> Self {
        let base_state = array![[Complex::new(1.0, 0.0)]];

        let state_matrix = input_bits
            .iter()
            .map(math::to_qbit_vector)
            .fold(base_state, |a, b| linalg::kron(&a, &b));

        Self {
            state: state_matrix,
        }
    }

    pub fn apply<'a, const ARITY: usize>(&mut self, op: &Operation<ARITY>) -> &mut Self {
        let target = op.targets()[0];
        let num_matrices = N + 1 - op.targets().len();

        let get_matrix = |i| {
            if i == target {
                return op.matrix();
            } else {
                return operation::identity(0).matrix();
            }
        };

        let base_state = array![[Complex::new(1.0, 0.0)]];

        let stage_matrix = (0..num_matrices)
            .map(get_matrix)
            .fold(base_state, |a, b| linalg::kron(&a, &b));

        self.state = stage_matrix.dot(&self.state);

        return self;
    }

    pub fn measure(&mut self, target: usize) -> bool {
        let mut prob_1 = 0.0;
        let mut prob_0 = 0.0;

        for (i, s) in self.state.iter().enumerate() {
            let prob = s.norm_sqr();

            if ((i >> target) & 1) == 1 {
                prob_1 += prob;
            } else {
                prob_0 += prob;
            }
        }

        let mut rng = rand::thread_rng();
        let x: f64 = rng.gen();

        let res = x > prob_0;

        let total_prob = if res { prob_1 } else { prob_0 };
        for (i, s) in self.state.iter_mut().enumerate() {
            if ((i >> target) & 1) != res as usize {
                *s = Complex::new(0.0, 0.0);
            } else {
                // TODO: Prove this v
                *s /= total_prob.sqrt();
            }
        }

        res
    }

    pub fn print_probabilities(&self) {
        for (i, s) in self.state.iter().enumerate() {
            println!("{:0N$b}: {}%", i, s.norm_sqr() * 100.0);
        }
    }
}

impl<const N: usize> PartialEq for Register<N> {
    fn eq(&self, other: &Self) -> bool {
        (&self.state - &other.state).iter().all(|e| e.norm() < 1e-8)
    }
}

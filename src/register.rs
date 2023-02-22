use crate::{
    math,
    operation::{self, Operation, OperationTrait},
};
use ndarray::{array, linalg, Array2};
use num::Complex;
use rand::prelude::*;

#[derive(Debug, PartialEq, Eq)]
pub enum OperationError {
    InvalidTarget{target: usize, n: usize},
    InvalidDimensions(usize, usize),
    NoTargets,
}

/// A quantum register containing N qubits.
#[derive(Clone, Debug)]
pub struct Register<const N: usize> {
    /// Represents the state of the quantum register as a vector with 2^N complex elements.
    ///
    /// The state is a linear combination of the basis vectors:
    /// |0..00>, |0..01>, |0..10>, ..., |1..11> (written in Dirac notation) which corresponds to the vectors:
    /// [1, 0, 0, ...]<sup>T</sup>, [0, 1, 0, ...]<sup>T</sup>, [0, 0, 1, ...]<sup>T</sup>, ..., [0, 0, ...., 0, 1]<sup>T</sup>
    ///
    /// In other words: state = a*|0..00> + b*|0..01> + c * |0..10> + ...
    ///
    /// The state vector is [a, b, c, ...]<sup>T</sup>, where |state_i|<sup>2</sup> represents the probability
    /// that the system will collapse into the state described by the ith basis vector.
    pub state: Array2<Complex<f64>>, // Should not be pub (it is pub now for testing purpouses)
}


impl<const N: usize> Register<N> {

    /// Creates a new state with an array of booleans with size N 
    pub fn new(input_bits: [bool; N]) -> Self {
        // Complex 1 by 1 identity matrix
        let base_state = array![[Complex::new(1.0, 0.0)]]; 
        
        // Creates state by translating bool to qubit
        // then uses qubits in tesnor product to create state
        let state_matrix = input_bits.iter()
            .map(math::to_qbit_vector)
            .fold(base_state, |a, b| linalg::kron(&b, &a));

        Self {
            state: state_matrix,
        }
    }
    /// Applys a quantum operation to the current state
    ///
    /// Input a state and an operation. Outputs the new state
    ///
    /// **Panics** if the operation is invalid or contains target bits
    /// outside of the valid range [0, N)
    pub fn apply<const ARITY: usize>(&mut self, op: &Operation<ARITY>) -> &mut Self {
        self.try_apply(op).expect("Could not apply operation")
    }

    /// Same as apply, except it returns an error instead of panicing when
    /// given incorrect arguments
    pub fn try_apply<const ARITY: usize>(&mut self, op: &Operation<ARITY>) -> Result<&mut Self, OperationError> {
        // Gets the target bit
        let target = *op.targets().first().ok_or(OperationError::NoTargets)?;

        // Check operation validity
        let expected_size = op.targets().len();
        let (rows, cols) = (op.matrix().shape()[0], op.matrix().shape()[1]);
        if (rows, cols) == (expected_size, expected_size) {
            return Err(OperationError::InvalidDimensions(rows, cols));
        }
        if target + ARITY > N {
            return Err(OperationError::InvalidTarget { target, n: N });
        }

        // Calculates the number of matrices in tensor product
        let num_matrices = N + 1 - op.targets().len();

        // If index i is equal to target bit returns matrix representation of operation
        // otherwise returns 2 by 2 identity matrix
        let get_matrix = |i| { if i == target { return op.matrix(); }
            else { return operation::identity(0).matrix(); }
        };

        // Complex 1 by 1 identity matrix
        let base_state = array![[Complex::new(1.0, 0.0)]];
        // Performs tensor product with the operation matrix and identity matrices
        let stage_matrix = (0..num_matrices)
            .map(get_matrix)
            .fold(base_state, |a, b| linalg::kron(&b, &a));

        // Calculates new state by performing dot product between current state and stage_matrix
        self.state = stage_matrix.dot(&self.state);

        return Ok(self);
    }

    /// Measure a quantum bit in the register and returns its measured value.
    ///
    /// Performing this measurement collapses the target qbit to either a one or a zero, and therefore
    /// modifies the state.
    ///
    /// The target bit specifies the bit which should be measured and should be in the range [0, N - 1].
    ///
    /// **Panics** if the supplied target is not less than the number of qubits in the register.
    pub fn measure(&mut self, target: usize) -> bool {
        self.try_measure(target).expect("Could not measure bit")
    }


    /// Same as measure, except it returns an error instead of panicing when given
    /// invalid arguments
    pub fn try_measure(&mut self, target: usize) -> Result<bool, OperationError> {
        // Validate arguments
        if target >= N { return Err(OperationError::InvalidTarget {target, n: N }); }

        let mut prob_1 = 0.0; // The probability of collapsing into a state where the target bit = 1
        let mut prob_0 = 0.0; // The probability of collapsing into a state where the target bit = 0

        for (i, s) in self.state.iter().enumerate() {
            // The probability of collapsing into state i
            let prob = s.norm_sqr();
            // If the target bit is set in state i, add its probability to prob_1 or prob_0 accordingly
            if ((i >> target) & 1) == 1 {
                prob_1 += prob;
            } else {
                prob_0 += prob;
            }
        }

        let mut rng = rand::thread_rng();
        let x: f64 = rng.gen();

        // The result of measuring the bit
        let res = x > prob_0;

        let total_prob = if res { prob_1 } else { prob_0 };
        for (i, s) in self.state.iter_mut().enumerate() {
            if ((i >> target) & 1) != res as usize {
                // In state i the target bit != the result of measuring that bit.
                // The probability of reaching this state is therefore 0.
                *s = Complex::new(0.0, 0.0);
            } else {
                // Because we have set some probabilities to 0 the state vector no longer
                // upholds the criteria that the probabilities sum to 1. So we have to normalize it.
                // Before normalization (state = X): sum(|x_i|^2) = total_prob
                // After normalization  (state = Y):  sum(|y_i|^2) = 1 = total_prob / total_prob
                // => sum((|x_i|^2) / total_prob) = sum(|y_i|^2)
                // => sum(|x_i/sqrt(total_prob)|^2) = sum(|y_i|^2)
                // => x_i/sqrt(total_prob) = y_i
                *s /= total_prob.sqrt();
            }
        }

        Ok(res)
    }

    /// Prints the probability in percent of falling into different states
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

#[cfg(test)]
mod tests {
    use crate::{operation, register::OperationError};

    use super::Register;

    #[test]
    fn invalid_target_returns_error() {
        let mut register = Register::new([false, false]);
        let res1 = register.try_apply(&operation::cnot(1, 0)).unwrap_err(); // Currently only the first parameter is used
        let _ = register.try_apply(&operation::cnot(0, 2)).unwrap(); // Currently only the first parameter is used

        assert_eq!(res1, OperationError::InvalidTarget{ target: 1, n: 2 });
    }
}

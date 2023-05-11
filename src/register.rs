//! The `register` module provides quantum register functionality.
use crate::{
    math::{self, c64},
    operation::{Operation, QuantumOperation},
};
use ndarray::{array, linalg, Array1, Array2};
use rand::prelude::*;
/// Errors which can occur when an operation is applied on the register.
#[derive(Debug, PartialEq, Eq)]
pub enum OperationError {
    /// Occurs when target is out of range or duplicate targets are given.
    InvalidTarget(usize),
    /// Occurs when an operation with invalid dimensions is given.
    InvalidDimensions(usize, usize),
    /// Occurs when an operation with an invalid arity is given.
    InvalidArity(usize),
    /// Occurs when a qubit is considered incorrect.
    InvalidQubit(usize),
}

/// A quantum register containing N qubits.
#[derive(Clone, Debug)]
pub struct Register {
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
    pub state: Array2<math::c64>, // Should not be pub (it is pub now for testing purpouses)
    size: usize,
}

impl Register {
    /// Creates a new state with an array of booleans with size N
    pub fn new(input_bits: &[bool]) -> Self {
        // Complex 1 by 1 identity matrix
        let base_state = array![[math::c64::new(1.0, 0.0)]];

        // Creates state by translating bool to qubit
        // then uses qubits in tesnor product to create state
        let state_matrix = input_bits
            .iter()
            .map(math::to_qbit_vector)
            .fold(base_state, |a, b| linalg::kron(&b, &a));

        Self {
            state: state_matrix,
            size: input_bits.len(),
        }
    }

    pub fn from_int(n: usize, k: usize) -> Self {
        Self {
            state: math::int_to_state(k, 1 << n),
            size: n,
        }
    }

    /// Creates a new state with a list of 2 dimensional arrays
    /// of complex numbers.
    ///
    /// Input qubits as [array![[Complex::new(1.0, 0.0)], array![Complex::new(0.0, 0.0)]]]
    /// Two qubits looks like this: [array![[Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0)]],
    ///                              array![[Complex::new(0.0, 0.0)], [Complex::new(1.0, 0.0)]]]
    ///
    /// **Panics** if 2 dimensional array doesn't contain 2 elements
    /// or if their probability doesn't add to 1
    pub fn from_qubits(input_bits: &[Array2<math::c64>]) -> Self {
        Self::try_from_qubits(input_bits).expect("Incorrect input qubits")
    }

    /// Tries to create a new state from list of qubits
    /// Returns a Result which is either a register or an
    /// error if input was not correct qubits
    pub fn try_from_qubits(input_bits: &[Array2<math::c64>]) -> Result<Self, OperationError> {
        //check if input is correct
        let res = input_bits.iter().map(Self::is_qubit);
        //check if everything was correct otherwise panic
        for (i, bool) in res.enumerate() {
            if !bool {
                return Err(OperationError::InvalidQubit(i));
            }
        }

        let base_state = array![[math::c64::new(1.0, 0.0)]];

        //create state
        let state_matrix = input_bits
            .iter()
            .fold(base_state, |a, b| linalg::kron(b, &a));

        Ok(Self {
            state: state_matrix,
            size: input_bits.len(),
        })
    }

    /// Checks if input qubit has total probability 1
    /// Outputs true if total probability is one and
    /// false if total probability is not one
    fn is_one(qubit: &Array2<math::c64>) -> bool {
        let mut total_prob = -1.0;
        //Create total prob by squaring individual values
        for values in qubit {
            total_prob += values.norm_sqr();
        }
        //Chech if total prob is 1
        total_prob.abs() < 0.000001
    }

    /// Checks that input qubit is correct
    /// Outputs false if the length of input is not 2
    /// otherwise outputs return value of is_one
    fn is_qubit(qubit: &Array2<c64>) -> bool {
        if qubit.len() == 2 {
            Self::is_one(qubit)
        } else {
            false
        }
    }

    /// Applys a quantum operation to the current state
    ///
    /// Input a state and an operation. Outputs the new state
    ///
    /// **Panics** if the operation is invalid or contains target bits
    /// outside of the valid range [0, N)
    pub fn apply(&mut self, op: &Operation) -> &mut Self {
        // self.try_apply(op).expect("Coult not apply operation")
        self.try_apply_fast(op).expect("Could not apply operation")
    }

    /// Tries to apply a quantum operation to the current state using fast Shrödinger simulation
    pub fn try_apply_fast(&mut self, op: &Operation) -> Result<&mut Self, OperationError> {
        let expected_size = op.targets().len();
        let (rows, cols) = (op.matrix().shape()[0], op.matrix().shape()[1]);
        if (rows, cols) == (expected_size, expected_size) {
            return Err(OperationError::InvalidDimensions(rows, cols));
        }
        if let Some(dup_target) = get_duplicate(&op.targets()) {
            return Err(OperationError::InvalidTarget(dup_target));
        }
        for target in op.targets() {
            if target >= self.size() {
                return Err(OperationError::InvalidTarget(target));
            }
        }

        let mut new_state = self.state.clone();
        let indecies =
            (0..1 << self.size()).filter(|x| op.targets().iter().all(|&y| (x >> y) & 1 == 0));
        //this loop should be possible to parallelize
        //loop over every index where the target bits are 0
        for index in indecies {
            //All targets bitpositions are 0
            //Find all permutations where the target bits are changed but all other bits are the same
            let mut permutations = Vec::new();
            //Loop over the number of possible permutations of the target bits
            for i in 0..1 << op.arity() {
                //set the base permutation to the current index
                let mut permutation = index;
                for (j, &target) in op.targets().iter().enumerate() {
                    //For each target bit, we need to check if the jth bit of i is 1
                    //if the jth bit of i is 1, then we need to flip the jth bit of the target
                    if (i >> j) & 1 == 1 {
                        permutation |= 1 << target;
                    }
                }
                permutations.push(permutation);
            }
            //create the array containing the values of the affected qubits
            let mut arr: Array1<c64> = Array1::zeros(1 << op.arity() as usize);
            for (i, &p) in permutations.iter().enumerate() {
                arr[i] = self.state[[(p) as usize, 0]];
            }
            //apply the operation to the affected qubits
            let res = op.matrix().dot(&arr);
            for (i, &p) in permutations.iter().enumerate() {
                new_state[[p as usize, 0]] = res[i];
            }
        }
        self.state = new_state;
        Ok(self)
    }

    pub fn try_apply(&mut self, op: &Operation) -> Result<&mut Self, OperationError> {
        // Check operation validity
        let expected_size = op.targets().len();
        let (rows, cols) = (op.matrix().shape()[0], op.matrix().shape()[1]);
        if (rows, cols) == (expected_size, expected_size) {
            return Err(OperationError::InvalidDimensions(rows, cols));
        }
        if let Some(dup_target) = get_duplicate(&op.targets()) {
            return Err(OperationError::InvalidTarget(dup_target));
        }
        for target in op.targets() {
            if target >= self.size() {
                return Err(OperationError::InvalidTarget(target));
            }
        }

        // Permutation indicating new order of qubits
        // Ex, perm[0]=3 means the qubit at idx 3 will be moved to idx 0
        let mut perm = op.targets();
        for i in 0..self.size {
            if !perm.contains(&i) {
                perm.push(i);
            }
        }

        // Create a copy of state and permute the qubits according to perm
        // Cloning here is not really necessary, all elements will be overwritten
        let mut permuted_state = self.state.clone();
        // Loop through and set each element in permuted_state
        for i in 0..permuted_state.len() {
            // Calculate the index j so that self.state[j] corresponds to permuted_state[i]
            // This is done by moving each bit in the number i according to perm
            let mut j: usize = 0;
            for (k, v) in perm.iter().enumerate().take(self.size) {
                // Copy the kth bit in i to the perm[k]th bit in j
                j |= ((i >> k) & 1) << v;
            }

            permuted_state[(i, 0)] = self.state[(j, 0)];
        }

        // Tensor product of operation matrix and identity
        let matrix = linalg::kron(&Array2::eye(1 << (self.size - op.arity())), &op.matrix());

        // Calculate new state
        permuted_state = matrix.dot(&permuted_state);

        // Permute back, similar to above but backwards (perm[k] -> k instead of the other way around)
        for i in 0..permuted_state.len() {
            let mut j: usize = 0;
            for (k, v) in perm.iter().enumerate().take(self.size) {
                j |= ((i >> v) & 1) << k;
            }
            self.state[(i, 0)] = permuted_state[(j, 0)];
        }

        Ok(self)
    }

    /// Apply a unary operation on all qubits.
    /// Returns a mutable reference to self.
    ///
    /// Input a unary operation.
    ///
    /// ***Panics*** if the arity of the operation is not 1.
    /// I.e. if the operation is not unary.
    pub fn apply_all(&mut self, operation: &Operation) -> &mut Self {
        self.try_apply_all(operation)
            .expect("Could not apply operation")
    }

    /// Tries to apply a unary operation on all qubits.
    /// Returns a Result with either a mutable reference to self or an error
    /// if the operation given is not unary.
    pub fn try_apply_all(&mut self, operation: &Operation) -> Result<&mut Self, OperationError> {
        // Return immediately if arity is not unary
        if operation.arity() != 1 {
            return Err(OperationError::InvalidArity(operation.arity()));
        }

        let matrix = (0..self.size).fold(array![[math::c64::new(1.0, 0.0)]], |acc, _| {
            linalg::kron(&acc, &operation.matrix())
        });

        // We dont need to do any swapping or target matching since the dimensions should always match if we
        // kron a unary operation reg.size() times.
        self.state = matrix.dot(&self.state);
        Ok(self)
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
        if target >= self.size() {
            return Err(OperationError::InvalidTarget(target));
        }

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
                *s = math::c64::new(0.0, 0.0);
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
        let n = self.size;
        for (i, s) in self.state.iter().enumerate() {
            println!("{:0n$b}: {}%", i, s.norm_sqr() * 100.0);
        }
    }

    /// Prints the state vector in binary representation.
    pub fn print_state(&self) {
        let n = self.size;
        for (i, s) in self.state.iter().enumerate() {
            println!("{i:0n$b}: {s}");
        }
    }

    /// Returns the number of qubits in the Register
    pub fn size(&self) -> usize {
        self.size
    }
}

impl PartialEq for Register {
    fn eq(&self, other: &Self) -> bool {
        (&self.state - &other.state).iter().all(|e| e.norm() < 1e-8)
    }
}

// Should probably be moved somewhere else
/// Returns a value which exists multiple times in the input vector, or None
/// if no such element exists
fn get_duplicate<T: Ord + Copy + Clone>(vec: &[T]) -> Option<T> {
    let mut vec_cpy = vec.to_vec();
    vec_cpy.sort_unstable();

    for i in 1..vec_cpy.len() {
        if vec_cpy[i] == vec_cpy[i - 1] {
            return Some(vec_cpy[i]);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use crate::{operation, register::OperationError};

    use super::Register;

    #[test]
    fn invalid_target_returns_error() {
        let mut register = Register::new(&[false, false]);
        let res1 = register.try_apply(&operation::cnot(1, 2)).unwrap_err(); // 2 is out of of bounds -> Error
        let _ = register.try_apply(&operation::cnot(1, 1)).unwrap_err(); // 1 == 1 -> Error
        let _ = register.try_apply(&operation::cnot(0, 1)).unwrap(); // Valid targets -> No error

        assert_eq!(res1, OperationError::InvalidTarget(2));
    }
}

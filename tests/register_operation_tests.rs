extern crate proptest;
use std::ops::Range;

use proptest::prelude::*;
use proptest::sample::{select, Select};
use quant::operation::{self, Operation};
use quant::register::Register;
use num::Complex;
use ndarray::{array, Array2, ArrayBase, OwnedRepr, Dim, linalg};
use std::{f64::consts, vec};

#[test]
// #[ignore = "Wait for feature confirmation"]
fn measure_on_zero_state_gives_false() {
    let mut register = Register::new(&[false]);
    let input = register.measure(0);
    let expected = false;

    assert_eq!(input, expected);
}

proptest!(
    #[test]
    fn test_not_on_arbitrary_qubit(i in 0..3) {
        let mut register = Register::new(&[false, false, false]);
        register.apply(&operation::not(i as usize));
        let input = register.measure(i as usize);
        let expected = true;
        assert_eq!(input, expected);
    }
    #[test]
    // #[ignore = "Indexing issue in register, is weird"]
    fn measure_four_qubits_gives_same_result(
        state in any::<bool>(),
        state2 in any::<bool>(),
        state3 in any::<bool>(),
        state4 in any::<bool>()
    ) {
        let mut register = Register::new(&[state, state2, state3, state4]);
        //                                  q0     q1       q2     q3
        let input = [
            register.measure(0),
            register.measure(1),
            register.measure(2),
            register.measure(3),
        ];
        let expected = [state, state2, state3, state4];
        assert_eq!(input, expected);
    }

    #[test]
    // #[ignore = "Indexing issue in register, is weird"]
    fn measure_eight_qubits_gives_same_result(
        state in any::<bool>(),
        state2 in any::<bool>(),
        state3 in any::<bool>(),
        state4 in any::<bool>(),
        state5 in any::<bool>(),
        state6 in any::<bool>(),
        state7 in any::<bool>(),
        state8 in any::<bool>()
    ) {
        let mut register = Register::new(&[state, state2, state3, state4, state5, state6, state7, state8]);
        let input = [
            register.measure(0),
            register.measure(1),
            register.measure(2),
            register.measure(3),
            register.measure(4),
            register.measure(5),
            register.measure(6),
            register.measure(7),
        ];
        let expected = [state, state2, state3, state4, state5, state6, state7, state8];
        assert_eq!(input, expected);
    }

    #[test]
    fn hadamard_hadamard_retains_original_state(i in 0..3) {
        let mut reg = Register::new(&[false,false,false]);
        let expected = reg.clone();

        let hadamard = operation::hadamard(i as usize);
        let input = reg.apply(&hadamard).apply(&hadamard);

        assert_eq!(*input, expected);
    }

    #[test]
    fn any_entangled_bit_measure_same(i in 0..6 as usize,mut j in 0..6 as usize){
        let mut reg = Register::new(&[false; 6]);
        if i == j {
            j = (j + 1) % 6;
        }
        let hadamard = operation::hadamard(j);
        let cnot = operation::cnot(i, j);

        // maximally entangle qubit i and j
        reg.apply(&hadamard);
        reg.apply(&cnot);

        // reg.print_probabilities();
        // this should measure index i and j
        assert_eq!(reg.measure(i), reg.measure(j));
    }

    #[test]
    // #[ignore = "Indexing issue in register, is weird"]
    fn first_bell_state_measure_equal(i in 0..5 as usize) {
        let mut reg = Register::new(&[false; 6]);
        let hadamard = operation::hadamard(i+1);
        let cnot = operation::cnot(i, i + 1);

        // maximally entangle qubit i and i + 1
        reg.apply(&hadamard);
        reg.apply(&cnot);

        reg.print_probabilities();
        // this should measure index i and i+1
        assert_eq!(reg.measure(i), reg.measure(i+1));
    }

    #[test]
    fn arbitrary_unary_applied_twice_gives_equal(op in UnaryOperation::arbitrary_with(0..6)) {
        let mut reg = Register::new(&[false; 6]);
        let expected = reg.clone();

        reg.apply(&op.0);
        reg.apply(&op.0);

        assert_eq!(reg, expected);

    }

    #[test]
    fn arbitrary_binary_applied_twice_gives_equal(op in BinaryOperation::arbitrary_with(0..6)) {
        let mut reg = Register::new(&[false; 6]);
        let expected = reg.clone();

        reg.apply(&op.0);
        reg.apply(&op.0);

        assert_eq!(reg, expected);
    }

    #[test]
    fn arbitrary_binary_applied_twice_gives_equal_after_swap_is_implemented(op in BinaryOperationAfterSwapIsImplemented::arbitrary_with(0..6)) {
        let mut reg = Register::new(&[false; 6]);
        let expected = reg.clone();

        reg.apply(&op.0);
        reg.apply(&op.0);

        assert_eq!(reg, expected);
    }


    #[test]
    #[ignore = "Apply does not figure out how to swap bits"]
    fn quantum_teleportation(q0 in StateVector::arbitrary_with(())) {
        let mut reg = Register::new(&[false;3]);



        let base_state = array![[Complex::new(1.0, 0.0)]]; 
        let expected = q0.clone();

        reg.state = [q0.0, to_qbit_vector(&false) , to_qbit_vector(&false)]
                    .iter()
                    .fold(base_state, |a, b| linalg::kron(&b, &a));
        // reg.state = 

        reg.apply(&operation::hadamard(2));
        reg.apply(&operation::cnot(2, 1));
        reg.apply(&operation::cnot(0, 1));
        reg.apply(&operation::hadamard(0));

        let c_0 = reg.measure(0);
        let c_1 = reg.measure(1);
        if c_1 {
            reg.apply(&operation::not(2));
        } 
        if c_0 {
            reg.apply(&operation::pauli_z(2));
        }
        let input = expected.clone(); //hur får jag tag i state vectorn för q2??

        println!("Expected: {:?}", expected.0);

        reg.print_probabilities();
        assert_eq!(expected, input);

    }

);
fn set_qubit_in_state(state: &mut ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 2]>>, index: usize, value: ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 2]>>) {
   for (i, s) in state.iter_mut().enumerate() {
        if (i >> index) & 1 == 1 {
            *s = *value.get((0,0)).unwrap();
        } else{
            *s = *value.get((1,0)).unwrap();
        }
   } 
}
pub fn to_qbit_vector(bit: &bool) -> Array2<Complex<f64>> {
    match bit {
        true => array![[Complex::new(0.0, 0.0)], [Complex::new(1.0, 0.0)]],
        false => array![[Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0)]],
    }
}


#[derive(Clone, Debug, PartialEq)]
struct StateVector(ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 2]>>);

impl Arbitrary for StateVector {
    type Parameters = ();
    type Strategy = Select<StateVector>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let frac = consts::FRAC_1_SQRT_2;
        select(vec!(
           StateVector(array![[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]]),
           StateVector(array![[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]]),
           StateVector(array![[Complex::new(frac, 0.0), Complex::new(frac, 0.0)]]),
           StateVector(array![[Complex::new(frac, 0.0), Complex::new(-frac, 0.0)]]),
           StateVector(array![[Complex::new(frac, 0.0), Complex::new(0.0, -frac)]]),
        ))
       
    }
}
fn real_to_complex(matrix: Array2<f64>) -> Array2<Complex<f64>> {
    matrix.map(|e| e.into())
}

#[derive(Debug, Clone)]
struct UnaryOperation(Operation);

impl Arbitrary for UnaryOperation {
    type Parameters = Range<usize>;
    type Strategy = Select<UnaryOperation>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let i = rand::thread_rng().gen_range(args);
        select(vec![
            UnaryOperation(operation::hadamard(i)),
            UnaryOperation(operation::not(i)),
            UnaryOperation(operation::identity(i)),
        ])
    }
}
#[derive(Debug, Clone)]
struct BinaryOperation(Operation);
impl Arbitrary for BinaryOperation {
    type Parameters = Range<usize>;
    type Strategy = Select<BinaryOperation>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let r = rand::thread_rng().gen_range(args.clone());
        let (i, j) = match args.max().unwrap() == r {
            true => (r, r - 1),
            false => (r + 1, r),
        };
        select(vec![
            BinaryOperation(operation::cnot(i, j)),
            BinaryOperation(operation::swap(i, j)),
        ])
    }
}
#[derive(Debug, Clone)]
struct BinaryOperationAfterSwapIsImplemented(Operation);
impl Arbitrary for BinaryOperationAfterSwapIsImplemented {
    type Parameters = Range<usize>;
    type Strategy = Select<BinaryOperationAfterSwapIsImplemented>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let i = rand::thread_rng().gen_range(args.clone());
        let mut j = rand::thread_rng().gen_range(args.clone());
        while i == j {
            j = rand::thread_rng().gen_range(args.clone());
        }
        select(vec![
            BinaryOperationAfterSwapIsImplemented(operation::cnot(i, j)),
            BinaryOperationAfterSwapIsImplemented(operation::swap(i, j)),
        ])
    }
}

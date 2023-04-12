extern crate proptest;
use std::ops::Range;

use ndarray::{array, linalg, Array2, ArrayBase, Dim, OwnedRepr};
use num::Complex;
use proptest::prelude::*;
use proptest::sample::{select, Select};
use quaru::math::{equal_qubits, to_qbit_vector};
use quaru::operation::{self, toffoli, Operation};
use quaru::register::Register;
use std::f64::consts;

#[test]
fn measure_on_zero_state_gives_false() {
    let mut register = Register::new(&[false]);
    let input = register.measure(0);
    let expected = false;

    assert_eq!(input, expected);
}

/// Tests that creating a state |0>, |1>, 1/sqrt(2)|0> + 1/sqrt(2)|1>
/// is equal to creating a state |0>, |1>, |0> and applying a hadamard
/// gate to qubit 2
#[test]
fn new_qubits_test() {
    let reg_qubits = Register::new_qubits(&[
        ndarray::array![[Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0)]],
        ndarray::array![[Complex::new(0.0, 0.0)], [Complex::new(1.0, 0.0)]],
        ndarray::array![
            [Complex::new(1.0 / (2.0_f64).sqrt(), 0.0)],
            [Complex::new(1.0 / (2.0_f64).sqrt(), 0.0)]
        ],
    ]);
    let mut reg = Register::new(&[false, true, false]);
    reg.apply(&operation::hadamard(2));
    assert_eq!(reg, reg_qubits);
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
        let hadamard = operation::hadamard(i);
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
    fn first_bell_state_measure_equal(i in 0..5_usize) {
        let mut reg = Register::new(&[false; 6]);
        let hadamard = operation::hadamard(i);
        let cnot = operation::cnot(i, i + 1);

        // maximally entangle qubit i and i + 1
        reg.apply(&hadamard);
        reg.apply(&cnot);

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
    fn swap_single_true_qubit(i in 0..5_usize, j in 0..5_usize){
        // qubit i is 1 and all other are 0
        // qubit i and j are swapped
        // qubit j should now be the only 1
        if i != j {
            let mut qubits = [false; 6];
            qubits[i] = true;
            let mut reg = Register::new(&qubits);
            let op = operation::swap(i, j);
            reg.apply(&op);

            for k in 0..5 {
                assert_eq!(reg.measure(k), k==j);
            }
        }
    }

    #[test]
    fn toffoli_test(n in 2..=6_usize,
        s1 in any::<bool>(),
        s2 in any::<bool>(),
        s3 in any::<bool>(),
        s4 in any::<bool>(),
        s5 in any::<bool>(),
        s6 in any::<bool>())
    {
        let init_state = [s1, s2, s3, s4, s5, s6];
        let mut reg = Register::new(&init_state);
        let init_target_value = init_state[n-1];

        reg.apply(&toffoli(&(0..n-1).collect(), n-1));

        let control_measure = (0..n-1).all(|i| reg.measure(i));
        let res = if control_measure {
            let target_measure = reg.measure(n-1);
            target_measure != init_target_value
        } else {
            let target_measure = reg.measure(n-1);
            target_measure == init_target_value
        };

        assert!(res);
    }

    #[test]
    fn apply_all_test(n in 2..=6_usize) {
        let mut reg1 = Register::new(&(0..n).map(|_| false).collect::<Vec<bool>>());
        let mut reg2 = Register::new(&(0..n).map(|_| false).collect::<Vec<bool>>());

        (0..reg1.size()).for_each(|i| { reg1.apply(&operation::hadamard(i)); });
        reg2.apply_all(&operation::hadamard(0));

        assert!(reg1 == reg2)
    }

    #[test]
    #[should_panic]
    fn apply_all_panics_if_not_unary(op in BinaryOperation::arbitrary_with(0..6)) {
        let mut reg = Register::new(&[false; 6]);
        reg.apply_all(&op.0);
    }

    #[test]
    fn quantum_teleportation(q0 in Qubit::arbitrary_with(())) {

        let expected = q0.0.clone();
        let mut reg = Register::new(&[false, false, true]);
        let base_state = array![[Complex::new(1.0, 0.0)]];
        let new_state = [expected.clone(), to_qbit_vector(&false) , to_qbit_vector(&false)]
                    .iter()
                    .fold(base_state, |a, b| linalg::kron(&b, &a));
        reg.state = new_state.clone();

        reg.apply(&operation::hadamard(2));
        reg.apply(&operation::cnot(2, 1));
        reg.apply(&operation::cnot(0, 1));
        reg.apply(&operation::hadamard(0));

        let c_1 = reg.measure(1);
        if c_1 {
            reg.apply(&operation::not(2));
        }
        let c_0 = reg.measure(0);
        if c_0 {
            reg.apply(&operation::pauli_z(2));
        }

        let input = get_state_of_qubit(reg.state.clone(), 2);
        let is_equal = equal_qubits(input.clone(), expected.clone());
        if !is_equal {
            reg.print_probabilities();
            println!("\n\n\nExpected: {:}", expected);
            println!("Got: {:}\n\n\n", input);
        }
        reg.print_probabilities();
        println!("input: {:}", input);
        assert!(is_equal);

    }

    #[test]
    fn superdense_coding(
        m0 in any::<bool>(),
        m1 in any::<bool>()
    ) {
        let message = [m0, m1];
        let mut register = Register::new(&[false; 2]);

        // Entangle qubit 0 and 1
        register.apply(&operation::hadamard(0));
        register.apply(&operation::cnot(0, 1));

        register.print_probabilities();
        println!();

        // Encode the message
        if message == [false, false] {
            register.apply(&operation::identity(0));
        } else if message == [false, true] {
            register.apply(&operation::not(0));
        } else if message == [true, false] {
            register.apply(&operation::pauli_z(0));
        } else if message == [true, true] {
            register.apply(&operation::pauli_z(0));
            register.apply(&operation::not(0));
        }

        register.print_probabilities();
        println!();

        // Decode message
        register.apply(&operation::cnot(0, 1));
        register.apply(&operation::hadamard(0));

        register.print_probabilities();
        println!();

        println!("Result: {}{}", register.measure(1) as i32, register.measure(0) as i32);
        assert_eq!(message, [register.measure(0), register.measure(1)]);
    }
);

pub fn get_state_of_qubit(
    state: ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 2]>>,
    n: usize,
) -> Array2<Complex<f64>> {
    let mut beta: Complex<f64> = Complex::new(0.0, 0.0);
    let mut alpha: Complex<f64> = Complex::new(0.0, 0.0);
    for (i, s) in state.iter().enumerate() {
        if (i >> n) & 1 == 1 {
            beta += s;
        } else {
            alpha += s;
        }
    }
    let prob = alpha.norm().powi(2) + beta.norm().powi(2);
    alpha /= prob.sqrt();
    beta /= prob.sqrt();
    array![[alpha], [beta]]
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qubit(Array2<Complex<f64>>);

impl Arbitrary for Qubit {
    type Parameters = ();
    type Strategy = Select<Qubit>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let frac = consts::FRAC_1_SQRT_2;
        select(vec![
            Qubit(array![[Complex::new(1.0, 0.0)], [Complex::new(0.0, 0.0)]]),
            Qubit(array![[Complex::new(0.0, 0.0)], [Complex::new(1.0, 0.0)]]),
            Qubit(array![[Complex::new(frac, 0.0)], [Complex::new(frac, 0.0)]]),
            Qubit(array![
                [Complex::new(frac, 0.0)],
                [Complex::new(-frac, 0.0)]
            ]),
            Qubit(array![
                [Complex::new(frac, 0.0)],
                [Complex::new(0.0, -frac)]
            ]),
        ])
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
        let i = rand::thread_rng().gen_range(args.clone());

        let mut j = rand::thread_rng().gen_range(args.clone());
        while i == j {
            j = rand::thread_rng().gen_range(args.clone());
        }

        select(vec![
            BinaryOperation(operation::cnot(i, j)),
            BinaryOperation(operation::swap(i, j)),
        ])
    }
}

extern crate proptest;

use std::ops::Range;

use proptest::prelude::*;
use proptest::sample::{select, Select};
use quaru::operation::{self, toffoli, Operation};
use quaru::register::Register;

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
    // #[ignore = "Indexing issue in register, is weird"]
    fn first_bell_state_measure_equal(i in 0..5 as usize) {
        let mut reg = Register::new(&[false; 6]);

        let hadamard = operation::hadamard(i);
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
    fn swap_single_true_qubit(i in 0..5 as usize, j in 0..5 as usize){
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
    fn toffoli_test(n in 2..=6 as usize,
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
            target_measure == !init_target_value
        } else {
            let target_measure = reg.measure(n-1);
            target_measure == init_target_value
        };

        assert!(res);
    }

);

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

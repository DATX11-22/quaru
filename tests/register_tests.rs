use proptest::prelude::*;
use quant::register::Register;

#[test]
fn measure_on_zero_state_gives_false() {
    let mut register = Register::new([false]);
    let input = register.measure(0);
    let expected = false;

    assert_eq!(input, expected);
}

proptest!(
    #[test]
    fn measure_four_qubits_gives_same_result(
        state in any::<bool>(),
        state2 in any::<bool>(),
        state3 in any::<bool>(),
        state4 in any::<bool>()
    ) {
        let mut register = Register::new([state, state2, state3, state4]);
        let input = [
            register.measure(3),
            register.measure(2),
            register.measure(1),
            register.measure(0),
        ];
        let expected = [state, state2, state3, state4];
        // let expected = [state4, state3, state2, state];
        assert_eq!(input, expected);
    }
);
proptest!(
    #[test]
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
        let mut register = Register::new([state, state2, state3, state4, state5, state6, state7, state8]);
        let input = [
            register.measure(7),
            register.measure(6),
            register.measure(5),
            register.measure(4),
            register.measure(3),
            register.measure(2),
            register.measure(1),
            register.measure(0),
        ];
        let expected = [state, state2, state3, state4, state5, state6, state7, state8];
        // let expected = [state8, state7, state6, state5, state4, state3, state2, state];
        assert_eq!(input, expected);
    }
);

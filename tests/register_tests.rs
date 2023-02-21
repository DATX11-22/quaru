use quant::register::Register;

#[test]
fn measure_on_zero_state_gives_false() {
    let mut register = Register::new([false]);
    let input = register.measure(0);
    let expected = false;

    assert_eq!(input, expected);
}


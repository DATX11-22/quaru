use quant::register::Register;


fn main() {
    let mut register = Register::new([true, false, false]);

    register.print_probabilities();

    println!("{}", register.measure(0));

    register.print_probabilities();
}

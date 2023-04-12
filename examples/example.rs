use quaru::{register::Register, operation};


fn main() {
    let mut register = Register::new(&[false, false]);

    register.print_probabilities();

    register.apply(&operation::hadamard(0));
    register.apply(&operation::cnot(0, 1));

    println!();
    register.print_probabilities();
}

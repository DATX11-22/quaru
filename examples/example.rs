use quant::register::Register;
use quant::operation::{Operation, self};


fn main() {
    let mut register = Register::new([true, false]);
    register.print_probabilities();
    println!();
    println!();
    println!();
    println!("State: {}", register.state);
    println!();
    println!();
    register.apply(&operation::not(0));
    // register.apply(&operation::hadamard(0));
    // register.apply(&operation::cnot(0, 1));
    // register.print_probabilities();
    println!("State: {}", register.measure(0));
    println!();
    println!();
    println!();
    register.print_probabilities();
}

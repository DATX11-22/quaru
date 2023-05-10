use quaru::register::Register;
use quaru::operation::*;

fn main() {
    // Create register with 2 qubits
    let mut register = Register::new(&[false; 2]);   

    // Apply unary hadamard operation
    register.apply(&hadamard(0));

    // Apply binary CNOT operation
    register.apply(&cnot(0, 1));

    // Print probabilities of all states
    register.print_probabilities();

    // Measure and print result
    println!("Measured: {}{}", register.measure(1) as usize, register.measure(0) as usize);
}


use quaru::{operation::*, register::*};


fn main() {
    let mut reg = Register::new(&[false; 2]);
    let mut circuit = QuantumCircuit::new();

    circuit.add_operation(Box::new(hadamard(0)));

    circuit.add_operation(Box::new(cnot(0, 1)));

    reg.apply_circuit(&circuit);

    reg.print_probabilities();





}
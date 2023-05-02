use quaru::{operation::*, register::*};

fn main() {
    let mut reg = Register::new(&[false; 2]);
    let mut circuit = QuantumCircuit::new();

    //apply optimizations
    circuit.reduce_circuit_cancel_gates();
    circuit.reduce_circuit_gates_with_same_targets();

    reg.apply_circuit(&mut circuit);

    reg.print_probabilities();
}

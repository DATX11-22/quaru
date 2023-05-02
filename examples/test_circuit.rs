
use quaru::{operation::*, register::*};


fn main() {
    let mut reg = Register::new(&[false; 2]);
    let mut circuit = QuantumCircuit::new();

    //Will cancel
    circuit.add_operation(hadamard(0));
    circuit.add_operation(hadamard(0));


    //Will multiply together
    circuit.add_operation(hadamard(0));
    circuit.add_operation(not(0));

    //will stay
    circuit.add_operation(phase(0));
    circuit.add_operation(phase(0));

    //will cancel
    circuit.add_operation(cnot(0, 1));
    circuit.add_operation(cnot(0, 1));


    circuit.reduce_circuit_cancel_gates();
    
    reg.apply_circuit(&mut circuit);

    reg.print_probabilities();





}
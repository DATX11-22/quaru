use quaru::{operation::{*, self}, register::*};

fn main() {
    let mut reg = Register::new(&[true, false, false]);

    let mut circuit = quantum_teleportation_circuit();

    //apply optimizations
    circuit.reduce_circuit_cancel_gates();
    circuit.reduce_circuit_gates_with_same_targets();

    reg.apply_circuit(&mut circuit);

    reg.print_probabilities();

    
}
fn quantum_teleportation_circuit() -> QuantumCircuit {
    let mut circ = QuantumCircuit::new();

    circ.add_operation(hadamard(2));
    circ.add_operation(cnot(2, 1));
    circ.add_operation(cnot(0, 1));
    circ.add_operation(hadamard(0));

    circ.add_measurement(1);
    circ.add_measurement(0);
    
    circ.add_conditional_operation(not(2), vec![(1, true)]);
    circ.add_conditional_operation(pauli_z(2), vec![(0, true)]);

    circ
    
}




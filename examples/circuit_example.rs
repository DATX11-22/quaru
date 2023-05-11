use quaru::{
    operation::*,
    register::*,
};
use stopwatch::Stopwatch;

fn main() {
    let mut reg = Register::new(&[false; 2]);
    let mut circ = QuantumCircuit::new();
    circ.add_operation(hadamard(0));
    circ.add_operation(cnot(0, 1));
    circ.reduce_gates_with_one_off_size(2);
    reg.apply_circuit(&mut circ);
    reg.print_probabilities();
    let size = 12 as usize;
    let mut reg = Register::new(&vec![false; size]);
    let mut circ = QuantumCircuit::new();
    println!("Before circuit");
    let sw = Stopwatch::start_new();
    for i in 0..size / 2 {
        circ.add_operation(hadamard(i * 2));
        circ.add_operation(cnot(i * 2, i * 2 + 1));
    }
    //timestamps are at 12 qubits
    // circ.reduce_non_overlapping_gates(); //825ms
    circ.reduce_gates_with_one_off_size(2); //650ms
    //Together : 920ms
    //Nothing :  1674ms
    reg.apply_circuit(&mut circ);
    let circ_time = sw.elapsed_ms();
    println!("After circuit: ");
    reg.print_nonzero_probabilities();
    println!("------------------");
    reg = Register::new(&vec![false; size]);
    let sw = Stopwatch::start_new();
    for i in 0..size / 2 {
        reg.apply(&hadamard(i * 2));
        reg.apply(&cnot(i * 2, i * 2 + 1));
    }
    let reg_time = sw.elapsed_ms();
    println!("After normal apply: ");
    reg.print_nonzero_probabilities();

    println!("Circuit time: {}ms", circ_time);
    println!("Register time: {}ms", reg_time);
}

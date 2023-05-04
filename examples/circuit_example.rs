use ndarray::Array2;
use quaru::{
    math::real_arr_to_complex,
    operation::{self, *},
    register::*,
};
use stopwatch::Stopwatch;

fn main() {
    let size = 10 as usize;
    let mut init_vec = Vec::<bool>::new();
    for i in 0..size {
        init_vec.push(false);
    }
    let mut reg = Register::new(&init_vec);

    let mut circ = QuantumCircuit::new();

    println!("Before circuit");
    let sw = Stopwatch::start_new();
    for i in 0..size / 2 {
        circ.add_operation(hadamard(i));
        circ.add_operation(cnot(i, size - i - 1));
    }
    circ.reduce_non_overlapping_gates();
    reg.apply_circuit(&mut circ);
    let circ_time = sw.elapsed_ms();
    println!("After circuit: ");
    reg.print_nonzero_probabilities();
    println!("------------------");


    reg = Register::new(&init_vec);
    let sw = Stopwatch::start_new();
    for i in 0..size / 2 {
        reg.apply(&hadamard(i));
        reg.apply(&cnot(i, size - i - 1));
    }
    let reg_time = sw.elapsed_ms();
    println!("After normal apply: ");
    reg.print_nonzero_probabilities();

    println!("Circuit time: {}ms", circ_time);
    println!("Register time: {}ms", reg_time);
}

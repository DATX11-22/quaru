use quaru::math::real_arr_to_complex;
use quaru::register::{Register, self};
use quaru::operation::*;
use ndarray::array;

fn example_register() -> Register {
    let mut register = Register::new(&[false; 3]);
    register.state = real_arr_to_complex(array![[0.05_f64.sqrt()], [0.1_f64.sqrt()], [0.2_f64.sqrt()], [0.3_f64.sqrt()], [0.0], [0.35_f64.sqrt()], [0.0], [0.0]]);
    register
}



fn main() {
    let mut register = example_register();

    // Print probabilities of all states
    register.print_probabilities();

    // Apply NOT on qubit 2
    register.apply(&not(2));

    // Print probabilities of all states
    register.print_probabilities();

    // Measure and print result
    println!("Measured: {}{}{}", register.measure(2) as usize, register.measure(1) as usize, register.measure(0) as usize);
}

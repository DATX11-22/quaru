use ndarray::array;
use num::Complex;
use quant::register::Register;


fn main() {
    let mut register = Register::new(&[false, true, true]);

    register.state = Complex::new(1.0 / f64::sqrt(4.0), 0.0) * array![
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];

    register.print_probabilities();

    println!("{}", register.measure(1));
    let mut register = Register::new(&[true, false, false]);

    register.print_probabilities();

    println!("{}", register.measure(0));

    register.print_probabilities();
}

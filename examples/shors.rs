use clap::Parser;
use colored::Colorize;
use log::debug;
use ndarray::{array, Array2};
use num::traits::Pow;
use quaru::math::{c64, limit_denominator, modpow, ComplexFloat};
use quaru::operation::{Operation, QuantumCircuit};
use quaru::{operation, register::Register};
use rand::Rng;
use std::f64::consts::{self, PI};
use stopwatch::Stopwatch;

include!("shors_functions.rs");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number to factorize
    #[arg(short, long, default_value_t = 15)]
    number: u32,

    /// Number of times to run the algorithm
    #[arg(long, default_value_t = 1)]
    n_times: u32,

    #[arg(long, default_value_t = false)]
    use_circuit: bool,

    /// Run fast or slow period-finder?
    #[arg(short, long, default_value_t = false)]
    fast: bool,
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    let number = args.number;
    let n_times = args.n_times;
    let use_circuit = args.use_circuit;
    let fast = args.fast;

    let mut runtimes = Vec::<i64>::new();
    for _ in 0..n_times {
        let sw = Stopwatch::start_new();
        println!("Running for N = {number}");

        // Find a factor with shor's algorithm
        let d1 = shors(number, fast, use_circuit);
        let d2 = number / d1;
        println!(
            "The factors of {} are {} and {}",
            number.to_string().green(),
            d1.to_string().blue(),
            d2.to_string().yellow()
        );
        let t = sw.elapsed_ms();
        if t > 1 {
            runtimes.push(t);
        }
        println!("Time elapsed: {} ms", t.to_string().blue());
        println!("------------------------------------")
    }

    if runtimes.len() == 0 {
        println!("No runs took more than 1 ms");
        return;
    }
    let avg = runtimes.iter().sum::<i64>() / runtimes.len() as i64;
    let min = runtimes.iter().min().unwrap();
    println!(
        "Total time: {} ms after {} runs",
        runtimes.iter().sum::<i64>().to_string().red(),
        n_times.to_string().red()
    );
    println!("Avarage time: {} ms", avg.to_string().bright_purple());
    println!("Best time: {} ms", min.to_string().green());
}

#[cfg(test)]
mod tests {
    use ndarray::{array, linalg, Array2};
    use quaru::math::c64;
    use quaru::math::{equal_qubits, modpow};
    use quaru::operation::Operation;
    use quaru::register::Register;

    #[test]
    fn period_finder_working() {
        for fast in [false, true] {
            // Try many combinations of a and N.
            // For each combination, find the period of f(x) = a^x mod N.
            for number in 2..(if fast { 15 } else { 6 }) {
                for a in 2..number {
                    if gcd::euclid_u32(number, a) != 1 {
                        // f(x) is not periodic if a and N share a factor since in
                        // that case f(0) = 1 but there are no other solutions to f(x) = 1
                        // (a has no multiplicative order modulo N).
                        continue;
                    }

                    // Calculate the period classically
                    let mut period = 1;
                    let mut a_pow = a;
                    while a_pow != 1 {
                        a_pow = a_pow * a % number;
                        period += 1;
                    }

                    // Find the period with the quantum algorithm 20 times
                    let mut ok = 0;
                    for _ in 0..20 {
                        // Quantum Period Finding is likely to find the period or a factor of the period
                        let r = super::find_period(number, a, fast);
                        if period % r == 0 {
                            ok += 1;
                        }
                    }

                    // Quantum Period Finding is probabilistic, so we can't expect it to always work.
                    // Good enough if 15 of the 20 tests were ok.
                    assert!(ok >= 15);
                }
            }
        }
    }

    #[test]
    fn shors_working() {
        // Test all composite numbers up to 15.
        // Only 15 is not caught by classical tests.
        for n in [4, 6, 8, 9, 10, 12, 14, 15] {
            let r = super::shors(n, true, false);
            assert!(n % r == 0 && 1 < r && r < n);
        }
    }

    #[test]
    fn test_u_gate() {
        let n = 4;
        // Try all combinations of modulus, base, i and input,
        // where input*base^(2^i) (mod modulus) is calculated.
        for modulus in 1..1 << n {
            for base in 0..1 << n {
                for i in 0..1 << n {
                    for input in 0..1 << n {
                        let mut reg = Register::from_int(n, input);

                        // Create the gate
                        let gate = super::u_gate((0..n).collect(), modulus, base, i);

                        // Apply the gate
                        reg.apply(&gate);

                        // Classical calculation of the expected result
                        let answer = Register::from_int(
                            n,
                            input * modpow(base, 1 << i as u32, modulus) as usize
                                % modulus as usize,
                        );

                        // Check that the result is correct
                        assert!(equal_qubits(reg.state, answer.state));
                    }
                }
            }
        }
    }

    /// Add the constant a to an integer in fourier space, mod 2^n.
    pub fn add_operation(a: usize, targets: Vec<usize>) -> Operation {
        let n = targets.len();
        let mut matrix = Array2::eye(1);
        for i in 0..n {
            // The transformation for the ith qubit.
            let em = array![
                [c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
                [
                    c64::new(0.0, 0.0),
                    c64::from_polar(1.0, std::f64::consts::PI / 2_f64.powi(i as i32) * a as f64)
                ]
            ];

            matrix = linalg::kron(&matrix, &em);
        }

        Operation::new(matrix, targets).expect("Failed to create add operation")
    }

    #[test]
    fn qft_add() {
        let n = 5;

        let qft_gate = super::qft(n).expect("Creation of qft failed");
        for a in 0..50 {
            for b in 0..50 {
                // Test that ADD(a, QFT(b)) = QFT(a+b)
                let mut reg1 = Register::from_int(n, b % (1 << n));
                reg1.apply(&qft_gate);
                reg1.apply(&add_operation(a, (0..n).collect()));

                let mut reg2 = Register::from_int(n, (a + b) % (1 << n));
                reg2.apply(&qft_gate);

                assert!(equal_qubits(reg1.state, reg2.state));
            }
        }
    }
}

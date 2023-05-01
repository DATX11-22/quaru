use clap::Parser;
use colored::Colorize;
use log::debug;
use ndarray::{array, Array2};
use num::traits::Pow;
use quaru::math::{c64, limit_denominator, modpow, ComplexFloat};
use quaru::operation::Operation;
use quaru::{operation, register::Register};
use rand::Rng;
use std::f64::consts::{self, PI};
use stopwatch::Stopwatch;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number to factorize
    #[arg(short, long, default_value_t = 15)]
    number: u32,

    /// Number of times to run the algorithm
    #[arg(long, default_value_t = 1)]
    n_times: u32,

    /// Run fast or slow period-finder?
    #[arg(short, long, default_value_t = false)]
    fast: bool
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    let number = args.number;
    let n_times = args.n_times;
    let fast = args.fast;

    let mut runtimes = Vec::<i64>::new();
    for _ in 0..n_times {
        let sw = Stopwatch::start_new();
        println!("Running for N = {}", number);

        // Find a factor with shor's algorithm
        let d1 = shors(number, fast);
        let d2 = number / d1;
        println!(
            "The factors of {} are {} and {}",
            number.to_string().green(),
            d1.to_string().blue(),
            d2.to_string().yellow()
        );
        let t = sw.elapsed_ms();
        runtimes.push(t);
        println!("Time elapsed: {} ms", t.to_string().blue());
        println!("------------------------------------")
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

/// Create a gate that multiplies its input by a^2^i mod N.
fn u_gate(targets: Vec<usize>, modulus: u32, a: u32, i: usize) -> operation::Operation {
    // Calculate a^2^i mod N
    let a_pow_mod: usize = modpow(a, 1 << i, modulus) as usize;

    debug!("a = {}, i = {}, mod = {}", a, i, modulus);
    debug!("a^2^i % mod = {}", a_pow_mod);

    // Create the function for the controlled u gate
    let func = |x: usize| -> usize { (x * a_pow_mod) % modulus as usize };
    // Create the gate
    let u_gate = operation::to_quantum_gate(&func, targets.clone());

    u_gate
}

/// Shor's algorithm
/// This algorithm finds a factor of the number N.
fn shors(number: u32, fast: bool) -> u32 {
    // Shor's algorithm doesn't work for even numbers
    if number % 2 == 0 {
        return 2;
    }

    // Shor's algorithm doesn't work for prime powers
    // Testing up to k=log2(N) is enough because if k > log2(N), 2^k > N.
    // (Since we already tested for even numbers, testing up to log3(N) would actually be enough)
    for k in 2..number.ilog2() + 1 {
        let c = ((number as f64).powf((k as f64).recip()) + 1e-9) as u32;
        if c.pow(k) == number {
            // c^k = N, so c is a factor
            return c;
        }
    }

    let mut iter = 0;
    loop {
        if iter > 0 {
            debug!("");
        }
        debug!("=== Attempt {} ===", iter + 1);

        // Pick a random number 1 < a < number
        let a: u32 = rand::thread_rng().gen_range(2..number);

        debug!("Using a = {} as guess", a);

        // We are done if a and N share a factor
        let k = gcd::euclid_u32(a, number);
        if k != 1 {
            debug!("N and a share the factor {}, done", k);
            return k;
        }

        // Quantum part
        let r = find_period(number, a, fast);

        // We need an even r. If r is odd, try again.
        if r % 2 == 0 {
            // Calculate k = a^(r/2) % N
            let k = modpow(a, r / 2, number);

            // If a^(r/2) = -1 (mod N), try again
            if k != number - 1 {
                debug!("Calculated a^(r/2) % N = {}", k);

                let factor1 = gcd::euclid_u32(k - 1, number);
                let factor2 = gcd::euclid_u32(k + 1, number);
                debug!("GCD({}-1, N) = {}", k, factor1);
                debug!("GCD({}+1, N) = {}", k, factor2);

                // At this point, both factor1 and factor2 are guaranteed to be
                // non-trivial factors of N, so we can return either one.
                // It is also be the case that factor1 * factor2 = N.

                return factor1;
            } else {
                // We end up here if {gcd(k-1,N), gcd(k+1,N)} = {1, N}.
                debug!("a^(r/2) = -1 (mod N), trying again");
            }
        } else {
            debug!("r odd, trying again.")
        }

        iter += 1;
    }
}

fn find_period(number: u32, a: u32, fast: bool) -> u32 {
    if fast {
        find_period_fast(number, a)
    }
    else {
        find_period_slow(number, a)
    }
}

/// Calculate r, a good guess for the period of f(x) = a^x mod N.
fn find_period_slow(number: u32, a: u32) -> u32 {
    debug!("Running slow period finding");

    // We need n qubits to represent N
    let n = ((number + 1) as f64).log2().ceil() as usize;

    // Create a register with 3n qubits
    let mut reg = Register::new(&vec![false; 3 * n]);

    // Apply hadamard transform to the first 2n qubits
    let hadamard_transform = operation::hadamard_transform((0..2 * n).collect());
    debug!("Applying hadamard transform");
    reg.apply(&hadamard_transform);

    // Apply not to the first of the last n qubits, in order to create a "1" in the last n qubits
    debug!("Applying not");
    reg.apply(&operation::not(2 * n));

    // The so-called U gate calculates the product of its input with a^2^i mod N
    debug!("Applying U gates");
    // The U-gates are applied to the last n qubits
    let targets: Vec<usize> = (2 * n..3 * n).collect();
    for i in 0..2 * n {
        let u_gate = u_gate(targets.clone(), number, a, i);
        // There are 2n U gates, each controlled by one of the first 2n qubits
        let c_u_gate = operation::to_controlled(u_gate, i);

        debug!("Applying c_u_gate for i = {}", i);
        reg.apply(&c_u_gate);
    }

    // Apply the qft (Quantum Fourier Transform) to the first 2n qubits
    let qft = qft(2 * n);
    debug!("Applying qft");
    reg.apply(&qft.expect("Creation of qft failed"));

    // Measure the first 2n qubits and convert the results to an integer
    let mut res = 0;
    debug!("Measuring");
    for i in 0..2 * n {
        let m = if reg.measure(i) { 1 } else { 0 };
        res |= m << i;
    }
    debug!("res = {}", res);

    let theta = res as f64 / 2_f64.pow((2 * n) as f64);
    debug!("theta = {}", theta);
    // At this point, theta ≃ s/r, where s is a random number between 0 and r-1,
    // and r is the period of a^x (mod N).

    // Find the fraction s/r closest to theta with r < N (we know the period is less than N).
    let r = limit_denominator(res, 2_u32.pow(2 * n as u32) - 1, number - 1).1;

    r
}

// See https://arxiv.org/pdf/quant-ph/0001066.pdf
fn find_period_fast(number: u32, a: u32) -> u32 {
    debug!("Running fast period finding");

    // Number of (qu)bits in `number`
    let n = ((number + 1) as f64).log2().ceil() as usize;

    // Register with n+1 qubits
    // Qubit 0 is the control qubit and qubits 1-n are the targets.
    let mut reg = Register::new(&vec![false; n + 1]);

    // Target qubits should represent 1 in binary
    debug!("Applying not");
    reg.apply(&operation::not(1));

    // All measurements as an integer
    let mut res = 0;

    // The sum in the formula for the R-gate (Parker&Plenio fig. 2 caption)
    // (exponentially weighted sum of measurements)
    let mut sm = 0.0;

    // Target qubits for the C-U-gates
    let targets: Vec<usize> = (1..n+1).collect();

    for i in 0..2 * n {
        // Put the control qubit in superposition
        reg.apply(&operation::hadamard(0));

        // Create and apply C-U-gate
        let u_gate = u_gate(targets.clone(), number, a, 2*n-1-i);
        let c_u_gate = operation::to_controlled(u_gate, 0);
        debug!("Applying c_u_gate for i = {}", i);
        reg.apply(&c_u_gate);
        
        // Now need to do a slice of QFT for the control qubit.
        let phi = c64::new(0.0, -2.0 * PI * sm).exp();
        let r_gate = Operation::new(
            array![
                [c64::new(1.0, 0.0), c64::new(0.0, 0.0)],
                [c64::new(0.0, 0.0), phi]
            ],
            vec![0]
        ).expect("Failed to construct R gate");
        debug!("Applying R- and H-gates to control qubits");
        reg.apply(&r_gate);
        reg.apply(&operation::hadamard(0));

        // Measure the control qubit.
        debug!("Measuring");
        let m = reg.measure(0);
        debug!("Measured {}", m);
        sm /= 2.0; // The sum is exponentially decaying
        if m {
            // Reset the control qubit to a known state of 0 to be reused
            // in the next iteration.
            debug!("Resetting control qubit to 0");
            reg.apply(&operation::not(0));

            res |= 1 << i;
            sm += 0.25;
        }
    }
    debug!("res = {}", res);

    let theta = res as f64 / 2_f64.pow((2 * n) as f64);
    debug!("theta = {}", theta);
    // At this point, theta ≃ s/r, where s is a random number between 0 and r-1,
    // and r is the period of a^x (mod N).

    // Find the fraction s/r closest to theta with r < N (we know the period is less than N).
    let r = limit_denominator(res, 2_u32.pow(2 * n as u32) - 1, number - 1).1;

    r
}

/// Returns the Quantum Fourier Transformation gate for the first n qubits in the register
pub fn qft(n: usize) -> Option<Operation> {
    let m = 1 << n;
    let mut matrix = Array2::zeros((m, m));
    let w = consts::E.powc(c64::new(0.0, 2.0 * consts::PI / m as f64));
    for i in 0..m as i32 {
        for j in 0..m as i32 {
            matrix[(i as usize, j as usize)] = w.powi(i * j) * (1.0 / (m as f64).sqrt());
        }
    }
    Operation::new(matrix, (0..n).collect())
}

#[cfg(test)]
mod tests {
    use quaru::math::{equal_qubits, modpow};
    use quaru::register::Register;

    #[test]
    fn period_finder_working() {
        for fast in [false, true] {
            // Try many combinations of a and N.
            // For each combination, find the period of f(x) = a^x mod N.
            for number in 2..(if fast {15} else {6}) {
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
            let r = super::shors(n, true);
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
}


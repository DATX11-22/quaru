use clap::Parser;

use colored::Colorize;
use log::debug;
use num::traits::Pow;
use quaru::{operation, register::Register};
use quaru::math::{limit_denominator, modpow};
use rand::Rng;
use stopwatch::Stopwatch;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number to factorize
    #[arg(short, long, default_value_t = 15)]
    number: u32,

    /// Number of times to run the algorithm
    #[arg(short, long, default_value_t = 1)]
    n_times: u32,
}

fn main() {
    let args = Args::parse();
    let number = args.number;
    let n_times = args.n_times;

    let mut runtimes = Vec::<i64>::new();
    for _ in 0..n_times {
        let sw = Stopwatch::start_new();
        println!("Running for N = {}", number);

        // Find a factor with shor's algorithm
        let d1 = shors(number);
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
    let a_pow_mod: i32 = modpow(a, 1<<i, modulus) as i32;

    debug!("a = {}, i = {}, mod = {}", a, i, modulus);
    debug!("a^2^i % mod = {}", a_pow_mod);

    // Create the function for the controlled u gate
    let func = |x: i32| -> i32 { (x * a_pow_mod) % modulus as i32 };
    // Create the gate
    let u_gate = operation::to_quantum_gate(&func, targets.clone());

    u_gate
}

/// Shor's algorithm
/// This algorithm finds a factor of the number N.
fn shors(number: u32) -> u32 {
    // Shor's algorithm doesn't work for even numbers
    if number % 2 == 0 {
        return 2;
    }

    // Shor's algorithm doesn't work for prime powers
    for k in 2..number.ilog2()+1 {
        let c = ((number as f64).powf((k as f64).recip()) + 1e-9) as u32;
        if c.pow(k) == number {
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
        let r = find_r(number, a);

        if r % 2 == 0 {
            // Calculate a^(r/2) % N
            let k = modpow(a, r / 2, number);

            if k != number - 1 {
                debug!("Calculated a^(r/2) % N = {}", k);

                let factor1 = gcd::euclid_u32(k - 1, number);
                let factor2 = gcd::euclid_u32(k + 1, number);
                debug!("GCD({}-1, N) = {}", k, factor1);
                debug!("GCD({}+1, N) = {}", k, factor2);
                return factor1.max(factor2);
            } else {
                debug!("a^(r/2) = -1 (mod N), trying again");
            }
        } else {
            debug!("r odd, trying again.")
        }

        iter += 1;
    }
}

/// Finds r such that maybe a^(r/2) + 1 shares a factor with N
fn find_r(number: u32, a: u32) -> u32 {
    // We need n qubits to represent N
    let n = (number as f64).log2().ceil() as usize;

    // Create a register with 3n qubits
    let mut reg = Register::new(&vec![false; 3 * n]);

    // Apply hadamard transform to the first 2n qubits
    let hadamard_transform = operation::hadamard_transform((0..2 * n).collect());
    debug!("Applying hadamard transform");
    reg.apply(&hadamard_transform);

    // Apply not to the first of the last n qubits, in order to create a "1" in the last n qubits
    debug!("Applying not");
    reg.apply(&operation::not(2 * n));

    debug!("Applying U gates");
    // Targets for the controlled u gates
    let targets: Vec<usize> = (2 * n..3 * n).collect();
    for i in 0..2 * n {
        let u_gate = u_gate(targets.clone(), number, a, i);
        let c_u_gate = operation::to_controlled(u_gate, i);

        debug!("Applying c_u_gate for i = {}", i);
        reg.apply(&c_u_gate);
    }

    // Apply the qft
    let qft = operation::qft(2 * n);
    debug!("Applying qft");
    reg.apply(&qft);

    // Measure the first 2n qubits
    let mut res = 0;
    debug!("Measuring");

    for i in 0..2 * n {
        let m = if reg.measure(i) { 1 } else { 0 };
        res |= m << i;
    }
    debug!("res = {}", res);

    let theta = res as f64 / 2_f64.pow((2 * n) as f64);
    debug!("theta = {}", theta);

    let r = limit_denominator(res, 2_u32.pow(2 * n as u32) - 1, 1 << n).1;

    r
}

#[cfg(test)]
mod tests {
    use quaru::register::Register;
    use quaru::math::{equal_qubits, modpow};

    #[test]
    fn period_finder_working() {
        for number in 2..8 {
            for a in 2..number {
                if gcd::euclid_u32(number, a) != 1 {
                    continue;
                }
                let mut period = 1;
                let mut a_pow = a;
                while a_pow != 1 {
                    a_pow = a_pow * a % number;
                    period += 1;
                }

                let mut ok = 0;
                for _ in 0..20 {
                    let r = super::find_r(number, a);
                    if period % r == 0 {
                        ok += 1;
                    }
                }
                assert!(ok >= 15);
            }
        }
    }

    #[test]
    fn shors_working() {
        for n in [4,6,8,9,10,12,14,15] {
            let r = super::shors(n);
            assert!(n % r == 0 && 1 < r && r < n);
        }
    }

    #[test]
    fn test_u_gate() {
        let n = 4;
        for modulus in 1..1<<n {
            for base in 0..1<<n {
                for i in 0..1<<n {
                    for input in 0..1<<n {
                        let mut reg = Register::from_int(n, input);

                        let gate = super::u_gate((0..n).collect(), modulus, base, i);

                        reg.apply(&gate);

                        let answer = Register::from_int(n, input * modpow(base, 1<<i as u32, modulus) as usize % modulus as usize);

                        assert!(equal_qubits(reg.state, answer.state));
                    }
                }
            }
        }
    }
}

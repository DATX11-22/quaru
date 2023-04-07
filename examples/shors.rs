use std::env;

use colored::Colorize;
use log::debug;
use num::traits::Pow;
use quaru::{operation, register::Register};
use quaru::math::{limit_denominator, modpow};
use rand::Rng;
use stopwatch::Stopwatch;

fn main() {
    let args: Vec<String> = env::args().collect();
    let (number, n_times) = parse_args(args);

    let mut times = Vec::<i64>::new();
    for _ in 0..n_times {
        let sw = Stopwatch::start_new();
        println!("Running for N = {}", number);
        let d1 = shors(number);
        let d2 = number / d1;
        println!(
            "The factors of {} are {} and {}",
            number.to_string().green(),
            d1.to_string().blue(),
            d2.to_string().yellow()
        );
        let t = sw.elapsed_ms();
        times.push(t);
        println!("Time elapsed: {} ms", t.to_string().blue());
        println!("------------------------------------")
    }

    let avg = times.iter().sum::<i64>() / times.len() as i64;
    let min = times.iter().min().unwrap();
    println!(
        "Total time: {} ms after {} runs",
        times.iter().sum::<i64>().to_string().red(),
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

fn shors(number: u32) -> u32 {
    if number % 2 == 0 {
        return 2;
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

fn parse_args(args: Vec<String>) -> (u32, u32) {
    let number = args
        .iter()
        .find(|x| x.starts_with("N="))
        .unwrap_or(&"N=15".to_owned())
        .split("=")
        .last()
        .unwrap()
        .parse::<u32>()
        .unwrap();
    let n_times = args
        .iter()
        .find(|x| x.starts_with("n_times="))
        .unwrap_or(&"n_times=1".to_owned())
        .split("=")
        .last()
        .unwrap()
        .parse::<u32>()
        .unwrap();
    (number, n_times)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, s};
    use num::{abs, Complex};
    use quaru::operation::{self, Operation, OperationTrait};
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
                for i in 0..20 {
                    let r = super::find_r(number, a);
                    if period % r == 0 {
                        ok += 1;
                    }
                }
                assert!(ok >= 15);
            }
        }
    }

    fn is_prime(n: u32) -> bool {
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        for i in 3..n {
            if n % i == 0 {
                return false;
            }
        }
        true
    }

    #[test]
    fn shors_working() {
        for n in 2..16 {
            if is_prime(n) {
                continue;
            }
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

                        let mut answer = Register::from_int(n, input * modpow(base, 1<<i as u32, modulus) as usize % modulus as usize);

                        assert!(equal_qubits(reg.state, answer.state));
                    }
                }
            }
        }
    }
}

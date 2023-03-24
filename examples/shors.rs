use std::env;

use colored::Colorize;
use stopwatch::Stopwatch;
use quant::{operation, register::Register};
use rand::Rng;

fn main() { 
    let args: Vec<String> = env::args().collect();
    let (N, n_times, debug) = parse_args(args);

    let mut times = Vec::<i64>::new();
    for _ in 0..n_times {
        let sw = Stopwatch::start_new();
        println!("Running for N = {}", N);
        let d1 = shors(N, debug);
        let d2 = N / d1;
        println!(
            "The factors of {} are {} and {}",
            N.to_string().green(),
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
    println!("Avarage time: {} ms", 
             avg.to_string().bright_purple()
    );
    println!("Best time: {} ms", 
            min.to_string().green()
    );
}

fn modpow(mut base: u32, mut exponent: u32, modulus: u32) -> u32 {
    let mut result = 1;
    while exponent > 0 {
        if exponent % 2 == 1 {
            result = (result * base) % modulus;
        }
        exponent /= 2;
        base = base*base % modulus;
    }
    return result;
}

fn shors(N: u32, debug : bool) -> u32 {
    // KOLLA OM N ÄR JÄMNT HÄR

    let mut iter = 0;
    loop {
        if debug {
            println!("\n=== Attempt {} ===", iter+1);
        }

        // Pick a random number 1 < a < N
        let a: u32 = rand::thread_rng().gen_range(2..N);

        if debug {
            println!("Using a = {} as guess", a);
        }

        // We are done if a and N share a factor
        let k = gcd::euclid_u32(a, N);
        if k != 1 {
            if debug {
                println!("N and a share the factor {}, done", k)
            }
            return k;
        }

        // Quantum part
        let r = find_r(N, a, debug);

        if r % 2 == 0 {
            // Calculate a^(r/2) % N
            let k = modpow(a, r/2, N);

            if k != N-1 {
                if debug {
                    println!("Calculated a^(r/2) % N = {}", k);
                }

                let factor1 = gcd::euclid_u32(k - 1, N);
                let factor2 = gcd::euclid_u32(k + 1, N);
                if debug {
                    println!("GCD({}-1, N) = {}", k, factor1);
                    println!("GCD({}+1, N) = {}", k, factor2);
                }
                return factor1.max(factor2);
            }
            else if debug{
                println!("a^(r/2) = -1 (mod N), trying again");
            }
        }
        else if debug {
            println!("r odd, trying again.")
        }

        iter += 1;
    }
}

/// Finds r such that maybe a^(r/2) + 1 shares a factor with N
fn find_r(N: u32, a: u32, debug : bool) -> u32 {
    // We need n qubits to represent N
    let n = (N as f64).log2().ceil() as usize;

    // Create a register with 3n qubits
    let mut reg = Register::new(&vec![false; 3*n]);

    // Apply hadamard transform to the first 2n qubits
    let hadamard_transform = operation::hadamard_transform((0..2*n).collect());
    if debug {
        println!("Applying hadamard transform");
    }
    reg.apply(&hadamard_transform);

    // Apply not to the first of the last n qubits, in order to create a "1" in the last n qubits
    if debug {
        println!("Applying not");
    }
    reg.apply(&operation::not(2*n));

    if debug {
        println!("Applying U gates");
    }
    // Targets for the controlled u gates
    let targets: Vec<usize> = (2*n..3*n).collect();
    for i in 0..2*n{
        // Calculate 2^i
        let pow = 1<<i;

        // Find a^pow mod N
        let a_pow_mod: i32 = modpow(a, pow, N) as i32;

        if debug {
            println!("a = {}, pow = {}, mod = {}", a, pow, N);
            println!("a_pow_mod = {}", a_pow_mod);
        }

        // Create the function for the controlled u gate
        let func = |x: i32| -> i32 {
            (x * a_pow_mod) % N as i32
        };
        // Create the controlled u gate
        let u_gate = operation::to_quantum_gate(&func, targets.clone());
        let c_u_gate = operation::to_controlled(u_gate, i);
        if debug {
            println!("Applying c_u_gate for i = {}", i);
        }
        reg.apply(&c_u_gate);
    }

    // Apply the qft
    let qft = operation::qft(2_i32.pow(2*n as u32) as usize); 
    if debug {
        println!("Applying qft");
    }
    reg.apply(&qft);
    
    // Measure the first 2n qubits
    let mut res = 0;
    if debug {
        println!("Measuring");
    }

    for i in 0..2*n {
        let m = if reg.measure(i) {1} else {0};
        res |= m<<i;
    }
    if debug {
        println!("res = {}", res);
    }
    
    // find r from the first 2n qubits

    let dem = 2_i32.pow(2*n as u32) as u32;
    let ret =  dem / gcd::euclid_u32(res, dem);
    if debug {
        println!("r = {}", ret);
    }
    ret

}

fn parse_args(args: Vec<String>) -> (u32, u32, bool) {
    let N = args
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
    let debug = args
        .iter()
        .find(|x| x.starts_with("debug="))
        .unwrap_or(&"debug=true".to_owned())
        .split("=")
        .last()
        .unwrap()
        .parse::<bool>()
        .unwrap();   
    (N, n_times, debug)
}
   
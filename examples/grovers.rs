use clap::{arg, Parser};
use num::traits::Pow;
use quaru::{
    operation::{cz, hadamard, not, oracle_operation},
    register::Register,
};
use std::{
    f64::consts::PI,
    fmt::Display,
    time::{Duration, Instant},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// State to search for (decimal)
    #[arg(short, long)]
    target: usize,
}

fn main() {
    let args = Args::parse();

    let target = args.target;
    let regsize = ((target+1) as f64).log2().ceil() as usize;
    println!("Regsize: {}", regsize);

    let result = grovers_algorithm(target, regsize);

    println!("{}", result);
}

/// Grover's Algorithm
/// This algorhitm can perform a search in an unstructured list with a
/// complexity of O(sqrt(N)) utilizing quantum superposition.
/// The algorithm consists of several parts.
/// Initially, all of the states are put in superposition.
/// Oracle functions are then used to amplify the probability of the
/// "correct" state while reducing the others.
/// This should be repeated O(sqrt(N)) times, before collapsing the
/// state hopefully resulting in the target state.

struct GroversResult {
    target: usize,
    result: Vec<bool>,
    iterations: usize,
    time: Duration,
}

impl Display for GroversResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Grover's Algorithm executed with the following results:
Target: {}
Result: {} ({:?})
Iterations of oracle + diffuser: {}
Time elapsed: {} ms",
            self.target,
            to_decimal(&self.result),
            self.result,
            self.iterations,
            self.time.as_millis()
        )
    }
}

fn grovers_algorithm(winner: usize, regsize: usize) -> GroversResult {
    let mut reg = Register::new(&(0..regsize).map(|_| false).collect::<Vec<bool>>());

    // Start benchmark
    let start = Instant::now();

    // Start with creating a uniform superposition
    (0..reg.size()).for_each(|i| {
        reg.apply(&hadamard(i));
    });

    // Generate oracle matrix
    let oracle = oracle_operation(reg.size(), winner);

    // Calculate numbers of repetitions needed
    let n: usize = iterations(reg.size());
    for _ in 0..n {
        // U_f (oracle function) reflects the "winner" state
        reg.apply(&oracle);

        // U_s (diffuser) reflects back and amplifies the "winner" states
        diffuser(&mut reg);
    }

    // Measure
    let measured_state: Vec<bool> = (0..reg.size()).rev().map(|i| reg.measure(i)).collect();

    // Stop benchmark
    let time_elapsed = start.elapsed();

    GroversResult {
        target: winner,
        result: measured_state,
        iterations: n,
        time: time_elapsed,
    }
}

// The diffuser function U_s reflects about the average amplitude
// amplifying the target state while reducing the amplitude of
// other states.
fn diffuser(reg: &mut Register) {
    (0..reg.size()).for_each(|i| {
        reg.apply(&hadamard(i));
    });
    (0..reg.size()).for_each(|i| {
        reg.apply(&not(i));
    });
    reg.apply(&cz(&(0..reg.size() - 1).collect(), reg.size() - 1));
    (0..reg.size()).for_each(|i| {
        reg.apply(&not(i));
    });
    (0..reg.size()).for_each(|i| {
        reg.apply(&hadamard(i));
    });
}

// Calculates the optimal number of iterations needed for U_sU_f
// to get good amplitudes for the target states.
fn iterations(search_space: usize) -> usize {
    (PI / 4.0 * (2.0 as f64).pow(search_space as f64).sqrt() / 1.0).floor() as usize
}

fn to_decimal(arr: &[bool]) -> usize {
    let mut dec = 0;
    for (i, n) in arr.iter().rev().enumerate() {
        dec += if *n { 2_usize.pow(i as u32) } else { 0 };
    }
    dec
}

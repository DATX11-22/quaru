use clap::{arg, Parser};
use ndarray::Array2;
use num::traits::Pow;
use quaru::{
    operation::{cz, hadamard, not, Operation},
    register::Register, math::real_to_complex
};
use std::{
    f64::consts::PI,
    fmt::Display,
    time::{Duration, Instant}
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
    println!("Regsize: {regsize}");

    let result = grovers_algorithm(target, regsize);

    println!("{result}");
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
    reg.apply_all(&hadamard(0));

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
    reg.apply_all(&hadamard(0));
    reg.apply_all(&not(0));
    reg.apply(&cz(&(0..reg.size() - 1).collect(), reg.size() - 1));
    reg.apply_all(&not(0));
    reg.apply_all(&hadamard(0));
}

pub fn oracle_operation(regsize: usize, winner: usize) -> Operation {
    let n: usize = 2_usize.pow(regsize as u32);
    let mut matrix: Array2<f64> = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        matrix.row_mut(i)[i] = if i == winner { -1.0 } else { 1.0 };
    }

    let op = Operation::new(real_to_complex(matrix), (0..regsize).collect()).expect("Could not create oracle operation");
    op
}

// Calculates the optimal number of iterations needed for U_sU_f
// to get good amplitudes for the target states.
fn iterations(search_space: usize) -> usize {
    (PI / 4.0 * 2.0_f64.pow(search_space as f64).sqrt() / 1.0).floor() as usize
}

fn to_decimal(arr: &[bool]) -> usize {
    let mut dec = 0;
    for (i, n) in arr.iter().rev().enumerate() {
        dec += if *n { 2_usize.pow(i as u32) } else { 0 };
    }
    dec
}

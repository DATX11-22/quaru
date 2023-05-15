use clap::{arg, Parser};
use ndarray::Array2;
use num::traits::Pow;
use quaru::{
    math::real_arr_to_complex,
    operation::{hadamard, Operation},
    register::Register,
};
use std::{
    collections::HashSet,
    f64::consts::PI,
    fmt::Display,
    time::{Duration, Instant},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// State to search for (decimal)
    #[arg(short, long)]
    target: Option<usize>,
    // List of elements to search in
    #[arg(short, long, value_delimiter = ',')]
    list: Option<Vec<usize>>,
    /// Statistics flag
    #[arg(short, long)]
    statistics: bool,
}

fn main() {
    let args = Args::parse();

    if !args.statistics && (args.target.is_none() || args.list.is_none()) {
        println!("Please provide a target integer with --target and a list with --list, or run in statistics mode with --statistics");
    } else if !args.statistics {
        let target = args.target.unwrap();
        let list = args.list.unwrap();
        let regsize = ((target + 1) as f64).log2().ceil() as usize;
        let regsize = if regsize < 2 { 2 } else { regsize }; // Minimum register size is 2.
        println!("Regsize: {regsize}");

        let result = grovers_algorithm(target, &list, regsize, iterations_floor);

        match result {
            None => println!("No result found"),
            Some(result) => println!("{result}"),
        }
    } else {
        statistics();
    }
}

fn statistics() {
    let iterations_fns = [
        iterations_ceil,
        iterations_floor,
        iterations_exact,
        iterations_ms,
    ];
    println!("Testing accuracy of Grover's algorithm for targets 1..100 with different iteration functions");
    println!("Order of functions: ceil, floor, exact, ms");

    for iteration_fn in iterations_fns {
        let mut correct: usize = 0;
        for i in 0..100 {
            let regsize = ((i + 1) as f64).log2().ceil() as usize;
            let regsize = if regsize < 2 { 2 } else { regsize }; // Minimum register size is 2.
            let list = (0..100).collect::<Vec<usize>>();
            let result = grovers_algorithm(i, &list, regsize, iteration_fn);
            
            match result {
                None => continue,
                Some(result) => {
                    if to_decimal(&result.result) > list.len() {
                        continue;
                    }

                    if list[to_decimal(&result.result)] == i {
                        correct += 1;
                    }
                }
            }
        }

        println!("Correctness {}", correct as f64 / 100.0);
    }
}

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

/// Grover's Algorithm
/// This algorhitm can perform a search in an unstructured list with a
/// complexity of O(sqrt(N)) utilizing quantum superposition.
/// The algorithm consists of several parts.
/// Initially, all of the states are put in superposition.
/// An amplitude amplification process then amplifies the probability of the
/// "correct" state while reducing the others in iterations.
fn grovers_algorithm(
    winner: usize,
    list: &Vec<usize>,
    regsize: usize,
    iterations_fn: fn(usize, usize) -> usize,
) -> Option<GroversResult> {
    let mut reg = Register::new(&(0..regsize).map(|_| false).collect::<Vec<bool>>());

    // Start benchmark
    let start = Instant::now();

    // Start with creating a uniform superposition
    reg.apply_all(&hadamard(0));

    // Generate oracle matrix
    let oracle = oracle_operation(reg.size(), winner, &list);

    // Calculate M, the number of occurences of winner in the list
    let m: usize = list.iter().filter(|&n| *n == winner).count();
    // Terminate if m == 0
    if m == 0 {
        return None;
    }

    // Calculate numbers of repetitions needed
    let n: usize = iterations_fn(reg.size(), m);
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

    Some(GroversResult {
        target: winner,
        result: measured_state,
        iterations: n,
        time: time_elapsed,
    })
}

// The diffuser function U_s reflects about the average amplitude
// amplifying the target state while reducing the amplitude of
// other states.
fn diffuser(reg: &mut Register) {
    reg.apply_all(&hadamard(0));
    reg.apply(&diffusion_operation(reg.size()));
    reg.apply_all(&hadamard(0));
}

/// The oracle function U_f reflects the "winner" state
///
/// ***Panics*** if the Operation fails to be created
pub fn oracle_operation(regsize: usize, winner: usize, list: &Vec<usize>) -> Operation {
    let n: usize = 2_usize.pow(regsize as u32);
    let mut matrix: Array2<f64> = Array2::<f64>::zeros((n, n));

    // Find the indexes of the winner state(s) in list
    let mut winner_indexes: HashSet<usize> = HashSet::new();
    for (i, item) in list.iter().enumerate() {
        if *item == winner {
            winner_indexes.insert(i);
        }
    }

    for i in 0..n {
        matrix.row_mut(i)[i] = if winner_indexes.contains(&i) {
            -1.0
        } else {
            1.0
        };
    }

    Operation::new(real_arr_to_complex(matrix), (0..regsize).collect())
        .expect("Could not create oracle operation")
}

/// Creates the diffuser operation U_s used in the diffuser function
///
/// ***Panics*** if the Operation fails to be created
pub fn diffusion_operation(regsize: usize) -> Operation {
    let n: usize = 2_usize.pow(regsize as u32);
    let mut matrix: Array2<f64> = Array2::<f64>::zeros((n, n));
    matrix.row_mut(0)[0] = 1.0;
    for i in 1..n {
        matrix.row_mut(i)[i] = -1.0;
    }

    Operation::new(real_arr_to_complex(matrix), (0..regsize).collect())
        .expect("Could not create diffuser operation")
}

/// Calculates the optimal number of iterations needed for U_sU_f
/// to get good amplitudes for the target states.
/// Floors the result.
fn iterations_floor(regsize: usize, m: usize) -> usize {
    (PI / 4.0 * (2.0_f64.pow(regsize as f64) / m as f64).sqrt()).floor() as usize
}

/// Calculates the optimal number of iterations needed for U_sU_f
/// to get good amplitudes for the target states.
/// Ceils the result.
fn iterations_ceil(regsize: usize, m: usize) -> usize {
    (PI / 4.0 * (2.0_f64.pow(regsize as f64) / m as f64).sqrt()).ceil() as usize
}

/// Calculates the optimal number of iterations needed for U_sU_f
/// to get good amplitudes for the target states.
/// Uses Microsofts formula.
/// https://learn.microsoft.com/en-us/azure/quantum/concepts-grovers
fn iterations_ms(regsize: usize, m: usize) -> usize {
    (PI / 4.0 * (2.0_f64.pow(regsize as f64) / m as f64).sqrt() - 0.5_f64).floor() as usize
}

/// Calculates the optimal number of iterations needed for U_sU_f
/// to get good amplitudes for the target states.
/// Uses Nielsen and Chuangs exact formula.
fn iterations_exact(regsize: usize, m: usize) -> usize {
    let theta = theta(regsize);
    let n = 2.0_f64.pow(regsize as f64);
    (f64::acos((m as f64 / n).sqrt()) / theta).round() as usize
}

/// Calculates the angle theta that represents half a rotation
/// of the state towards the target
fn theta(regsize: usize) -> f64 {
    let n = 2.0_f64.pow(regsize as f64);
    f64::asin((2.0 * (n - 1.0).sqrt()) / n)
}

/// Converts a boolean array to a decimal number
fn to_decimal(arr: &[bool]) -> usize {
    let mut dec = 0;
    for (i, n) in arr.iter().rev().enumerate() {
        dec += if *n { 2_usize.pow(i as u32) } else { 0 };
    }
    dec
}

#[cfg(test)]
mod tests {
    use quaru::{operation::hadamard, register::Register};

    /// Tests that the oracle_operation function flips the correct state
    /// and leaves the others unchanged
    #[test]
    fn test_oracle_operation() {
        for target in 0..20 {
            let regsize = ((target + 1) as f64).log2().ceil() as usize;
            let regsize = if regsize < 2 { 2 } else { regsize }; // Minimum register size is 2.
            let mut reg = Register::new(&(0..regsize).map(|_| false).collect::<Vec<bool>>());
            let oracle = super::oracle_operation(reg.size(), target, &(0..20).collect());
            reg.apply_all(&hadamard(0));
            reg.apply(&oracle);
            // Check that the state is flipped
            let target_state = reg.state.get((target, 0)).unwrap();
            assert!(target_state.re.is_sign_negative());
        }
    }

    /// Tests that the diffuser_operation function reflects about the average amplitude
    /// amplifying the target state while reducing the amplitude of other states.
    /// This is done by checking that the amplitude of the non-target state is lower
    /// than the target state after applying the diffuser.
    #[test]
    fn test_diffuser_operation() {
        for target in 0..20 {
            let regsize = ((target + 1) as f64).log2().ceil() as usize;
            let regsize = if regsize < 2 { 2 } else { regsize }; // Minimum register size is 2.
            let mut reg = Register::new(&(0..regsize).map(|_| false).collect::<Vec<bool>>());
            let oracle = super::oracle_operation(reg.size(), target, &(0..20).collect());
            reg.apply_all(&hadamard(0));
            reg.apply(&oracle);
            crate::diffuser(&mut reg);
            reg.print_state();
            // Check that the amplitude of the target state is higher than the others
            let target_state = reg.state.get((target, 0)).unwrap().norm();
            for i in 0..reg.size() {
                if i != target {
                    assert!(reg.state.get((i, 0)).unwrap().norm() < target_state);
                }
            }
        }
    }
}

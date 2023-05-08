use quaru::{operation, register::Register};
use std::time::Instant;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    let mut xs = vec![]; // measurements, xs[i-1] is the time with i qubits.
    for n in 1..=14 {
        // Measure time as the minimum of 10 trials.
        let mut min_time = f64::INFINITY;
        for _ in 1..10 {
            let mut reg = Register::from_int(n, 0);
            let start = Instant::now();
            reg.apply_all(&operation::hadamard(0));
            let time = start.elapsed().as_secs_f64();
            min_time = min_time.min(time);
        }
        
        xs.push(min_time);
        println!("{} {} a", n, min_time);
    }

    println!("\nFinding regression curve");

    // Finding regression curve on the form y = ab^x + c, where a,b,c are parameters.
    let mut p = [1e-9_f64, 4.0, 1e-7]; // p = [a,b,c]
    let mut best_r2 = f64::MAX;
    let mut it: i64 = 0;
    loop {
        // Make a random change to the parameters.
        let idx = rng.gen_range(0..3);
        let f = rng.gen_range(0.99..1.01);
        p[idx] *= f;
        
        // Calculate sum of squared errors after the change
        let mut r2 = 0.0;
        for i in 1..=xs.len() {
            r2 += (xs[i-1].log2() - (p[0]*p[1].powi(i as i32) + p[2]).log2()).powf(2.0);
        }
        
        if r2 < best_r2 {
            // If the change was an improvement, keep it...
            best_r2 = r2;
        }
        else{
            // ... otherwise undo the change.
            p[idx] /= f;
        }
        
        it += 1;
        if it%20000000 == 0 {
            println!("\n{}*{}^x+{}\nError: {}", p[0], p[1], p[2], best_r2);
        }
    }
}

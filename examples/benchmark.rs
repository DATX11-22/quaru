use quaru::{operation, register::Register};
use std::time::Instant;
use rand::Rng;

include!("shors.rs");

fn is_prime(n: u32) -> bool {
    for i in 2..n {
        if n % i == 0 {
            return false;
        }
    }
    true
}

fn bench_shors(){
    let attempts = 100;
    for n in 4..=3000 {
        if is_prime(n) {
            continue;
        }
        let mut sum_time = 0.0;
        for _ in 0..attempts {
            let start = Instant::now();
            shors(n, true);
            let time = start.elapsed().as_secs_f64();
            sum_time += time
        }
        
        let avg_time = sum_time/attempts as f64;
        println!("{} {}", n, avg_time);
    }
}

enum Accumulator {
    Min,
    Avg,
}

fn benchmark(func: fn(i32) -> f64, xs: Vec<i32>, attempts: usize, acc: Accumulator) -> Vec<f64> {
    let mut ys = vec![];
    for n in xs {
        let mut measurements = vec![];
        for _ in 0..attempts {
            measurements.push(func(n));
        }
        let tot = match acc {
            Accumulator::Min => measurements.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            Accumulator::Avg => measurements.iter().sum::<f64>()/attempts as f64,
        };
        ys.push(tot);
        println!("{} {} a", n, tot);
    }
    ys
}

fn main() {
    let mut rng = rand::thread_rng();

    fn bench_shors(n: i32) -> f64 {
        let start = Instant::now();
        shors(n as u32, true);
        start.elapsed().as_secs_f64()
    }

    fn bench_apply_all(n: i32) -> f64 {
        let mut reg = Register::from_int(n as usize, 0);
        let start = Instant::now();
        reg.apply_all(&operation::hadamard(0));
        start.elapsed().as_secs_f64()
    }

    let xs: Vec<i32> = (1..=14).collect();
    //let ys = benchmark(bench_shors, xs, 100, Accumulator::Avg);
    let ys = benchmark(bench_apply_all, xs, 100, Accumulator::Min);


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
        for i in 0..xs.len() {
            r2 += (ys[i].log2() - (p[0]*p[1].powi(xs[i]) + p[2]).log2()).powf(2.0);
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
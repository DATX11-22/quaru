use log::debug;
use ndarray::{array, Array2};
use num::traits::Pow;
use quaru::math::{c64, limit_denominator, modpow, ComplexFloat};
use quaru::operation::{Operation, QuantumCircuit};
use quaru::{operation, register::Register};
use rand::Rng;
use std::f64::consts::{self, PI};
use std::time::Instant;

include!("shors_functions.rs");

fn is_shorable(n: u32) -> bool {
    // Check if prime
    let mut prime = true;
    for i in 2..n {
        if n % i == 0 {
            prime = false;
        }
    }
    if prime {
        return false;
    }

    // Check if even
    if n % 2 == 0 {
        return false;
    }

    // Check if power
    for k in 2..n.ilog2() + 1 {
        let c = ((n as f64).powf((k as f64).recip()) + 1e-9) as u32;
        if c.pow(k) == n {
            // c^k = N, so c is a factor
            return false;
        }
    }

    true
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
        shors(n as u32, true, false);
        start.elapsed().as_secs_f64()
    }

    fn bench_apply_all(n: i32) -> f64 {
        let mut reg = Register::from_int(n as usize, 0);
        let start = Instant::now();
        reg.apply_all(&operation::hadamard(0));
        start.elapsed().as_secs_f64()
    }

    let xs: Vec<i32> = (2..=100).filter(|&x| is_shorable(x as u32)).collect();
    let ys = benchmark(bench_shors, xs.clone(), 100, Accumulator::Avg);
    let formula = |p: Vec<f64>, x: i32| (p[0]*p[1].pow((x as f64).log2()));
    let mut p = vec![1e-6, 3.0];

    /*
    let xs: Vec<i32> = (1..=14).collect();
    let ys = benchmark(bench_apply_all, xs.clone(), 100, Accumulator::Min);
    let formula = |p: Vec<f64>, x: i32| (p[0]*p[1].powi(x) + p[2]);
    let mut p = vec![1e-9_f64, 4.0, 1e-7];
    */

    println!("\nFinding regression curve");

    // Finding regression curve on the form y = ab^x + c, where a,b,c are parameters.
    let mut best_r2 = f64::MAX;
    let mut it: i64 = 0;
    loop {
        // Make a random change to the parameters.
        let idx = rng.gen_range(0..p.len());
        let f = rng.gen_range(0.99..1.01);
        p[idx] *= f;
        
        // Calculate sum of squared errors after the change
        let mut r2 = 0.0;
        for i in 0..xs.len() {
            r2 += (ys[i].log2() - formula(p.clone(), xs[i]).log2()).powf(2.0);
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
        if it%5000000 == 0 {
            //println!("\n{}*{}^x+{}\nError: {}", p[0], p[1], p[2], best_r2);
            println!("\n{:?}\nError: {}", p, best_r2);
        }
    }
}
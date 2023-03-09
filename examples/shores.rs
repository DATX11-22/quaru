use gcd::Gcd;
use std::env;

use stopwatch::{Stopwatch};
use num::{Complex, traits::Pow};
use quant::{operation, register::Register};

use ndarray::{Array2, ArrayBase, Data, DataMut};


fn main() { 
    let args: Vec<String> = env::args().collect();
    let N = args.iter().find(|x| x.starts_with("N="))
                        .unwrap_or(&"N=15".to_owned()).split("=").last()
                        .unwrap().parse::<u128>().unwrap();
    let n_times = args.iter().find(|x| x.starts_with("n_times="))
                        .unwrap_or(&"n_times=1".to_owned()).split("=").last()
                        .unwrap().parse::<u32>().unwrap();
    let debug = args.iter().find(|x| x.starts_with("debug="))
                        .unwrap_or(&"debug=true".to_owned()).split("=").last()
                        .unwrap().parse::<bool>().unwrap();





    let mut times = Vec::<i64>::new();
    for _ in 0..n_times {
        let sw = Stopwatch::start_new();
        println!("Running for N = {}", N);
        let d1 = shores(N, debug);
        let d2 = N / d1;
        println!("The factors of {} are {} and {}", N, d1, d2);
        let t = sw.elapsed_ms();
        times.push(t);
        println!("Time elapsed: {} ms", t);
    }
    println!();
    println!();
    println!();
    let avg = times.iter().sum::<i64>() / times.len() as i64;
    let min = times.iter().min().unwrap();
    println!("Total time: {} ms after {} runs", times.iter().sum::<i64>(), n_times);
    println!("Avarage time: {} ms", avg);
    println!("Min time: {} ms", min);
}

fn shores(N: u128, debug : bool) -> u128 {
    //shortcut
    // if N % 2 == 0 {
    //     // N is even
    //     return 2;
    // }
    loop {
        let mut a: u128 = rand::random();
        a = a % N;
        a = if a <= 1 { 2 } else { a };
        let K = gcd::euclid_u128(a, N);
        //shortcut
        // if K != 1 && K != N {
        //     return K;
        // }

        if debug {
            println!("Using a = {} as guess", a);
        }

        //quantum part
        let r = find_r(N, a , debug);

        if r % 2 == 0 {
            // r is even so we can find a factor

            let guess_div = gcd::euclid_u128(a.pow(r / 2) + 1, N);
            if N  % guess_div == 0 && guess_div != 1 && guess_div != N{
                return guess_div;
            }
            else if debug{
                println!("Guess: {} was wrong", guess_div);
            }
        }
    }
}

/// Finds r such that maybe a(r/2) + 1 is a factor of N
fn find_r(N: u128, a: u128, debug : bool) -> u32 {


    // we need n qubits to represent N
    let n = (N as f64).log2().ceil() as usize;
    //create a register with 3n qubits
    let mut reg = Register::new(&(0..3*n).map(|_| false).collect::<Vec<bool>>());

    //apply hadamard transform to the first 2n qubits
    let hadamard_transform = operation::hadamard_transform((0..(2*n as usize)).collect());
    if debug {
        println!("Applying hadamard transform");
    }
    reg.apply(&hadamard_transform);
    if debug {
        println!("Applied hadamard transform");
    }

    if debug {
        println!("applying not");
    }
    //apply not to the first of the last n qubits, in order to create a "1" in the last n qubits
    reg.apply(&operation::not(2*n));
    if debug {
        println!("applied not");
    }

    //targets for the controlled u gates
    let targets : Vec<usize> = (2*n..3*n).collect();
    for i in 0..2*n{

        let pow = 2_i32.pow(i as u32) as u32;
        // find a^pow mod N
        let mut a_pow_mod : i32 = 1; 
        for _ in 0..pow {
            a_pow_mod = (a_pow_mod * (a as i32)) % N as i32;
        }
        // let a_pow_mod =  (a_pow % N as i64) as i32;
        if debug {
            println!("a = {}, pow = {}, mod = {}", a, pow, N);
            println!("a_pow_mod = {}", a_pow_mod);
        }

        // create the function for the controlled u gate
        let func = |x: i32| -> i32 {
            (x * a_pow_mod) % N as i32
        };
        // create the controlled u gate
        let u_gate = operation::to_qunatum_gate(&func, targets.clone());
        let c_u_gate = operation::to_controlled(u_gate, i);
        if debug {
            println!("applying c_u_gate for i = {}", i);
        }
        reg.apply(&c_u_gate);
        if debug {
            println!("Applied c_u_gate for i = {}", i);
        }
    }
    // apply the qft
    let qft = operation::qft(2_i32.pow(2*n as u32) as usize); 
    if debug {
        println!("applying qft");
    }
    reg.apply(&qft);
    if debug {
        println!("applied qft");
    }
    
    // measure the first 2n qubits
    let mut res =0;
    if debug {
        println!("measuring");
    }

    for i in 0..2*n {
        let m = if reg.measure(i) {1} else {0};
        res |= m<<i;
    }
    if debug {
        println!("measured");
        println!("res = {}", res);
    }
    println!("res = {}", res);

    
    // find r from the first 2n qubits

    let dem = 2_i32.pow(2*n as u32) as u32;
    let ret =  dem / gcd::euclid_u32(res, dem);
    if debug {
        println!("r = {}", ret);
    }
    ret

}

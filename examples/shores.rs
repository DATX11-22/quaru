use gcd::Gcd;

use ndarray::array;
use num::{Complex, traits::Pow};
use quant::{operation, register::Register};

use ndarray::{Array2, ArrayBase, Data, DataMut};


fn main() {

    // let mut reg = Register::new(&[false; 3]);    
    // let not = operation::cnot(1, 2);
    // let cnot = operation::to_controlled(not, 0);
    // println!("{:}", cnot.matrix);

    // reg.apply(&cnot);
    // return;
    // let arr = array![[1.0, 2.0], [3.0, 4.0]];
    // let inv_arr = arr.inv().unwrap();
    // // let inv_arr = array![[1.0, 2.0], [3.0, 4.0]];
    // let id = arr.dot(&inv_arr);
    // println!("{:}", id);
    // return;


    // let mut reg = Register::new(&[false; 3]);
    // let add_one = |x: i32| -> i32 {
    //     x+1
    // };
    // let targets : Vec<usize> = (0..=1).collect();
    // let add_one_gate = operation::to_qunatum_gate(&add_one, targets);
    // println!("{:}", add_one_gate.matrix);
    // // println!("{:?}", add_one_gate);
    // // println!("{:?}", add_one_controlled);

    // for _ in 0..3 {
    //     reg.apply(&add_one_gate);
    //     println!("Applied add one gate");
    // }
    // reg.print_probabilities();

    // return;

        // let mut reg = Register::new(&[false; 4]);
        // let qft = operation::qft(4);
        // println!("{:}", qft.matrix);
        // reg.apply(&qft);
        // reg.print_probabilities();
        // return;


    let N : u32 = 15;
    let d1 = shores(N);
    let d2 = N / d1;
    println!("The factors of {} are {} and {}", N, d1, d2);

    // let qft = operation::qft(4);
    // println!("{:}", qft.matrix)
}

fn shores(N: u32) -> u32 {
    if N % 2 == 0 {
        // N is even
        return 2;
    }
    loop {
        let mut a: u32 = rand::random();
        a = a % N;
        a = if a <= 1 { 2 } else { a };
        // a = 7;
        let K = gcd::euclid_u32(a, N);
        if K != 1 && K != N {
            return K;
        }

        println!("a = {}", a);
        let r = find_r(N, a);
        if r % 2 == 0 {
            // r is even so we can find a factor
            let guess_div = gcd::euclid_u32(a.pow(r / 2) + 1, N);
            if N % guess_div == 0 && guess_div != 1 && guess_div != N{
                return guess_div;
            }
        }
    }
}

//TODO: implement this function
fn find_r(N: u32, a: u32) -> u32 {
    // if a == 7 && N == 15 {
    //     return 4;
    // }
    let n = (N as f64).log2().ceil() as usize;
    let mut reg = Register::new(&(0..3*n).map(|_| false).collect::<Vec<bool>>());
    let hadamard_transform = operation::hadamard_transform((0..(2*n as usize)).collect());
    println!("Applying hadamard transform");
    reg.apply(&hadamard_transform);
    println!("Applied hadamard transform");

    println!("applying not");
    // reg.apply(&operation::not((3*n)-1));
    reg.apply(&operation::not(2*n));
    // reg.print_nonzero_probabilities();

    println!("applied not");

    let targets : Vec<usize> = (2*n..3*n).collect();
    for i in 0..2*n{

        let pow = 2_i32.pow(i as u32) as u32;
        let mut a_pow_mod : i32 = 1; 
        for _ in 0..pow {
            a_pow_mod = (a_pow_mod * (a as i32)) % N as i32;
        }
        // let a_pow_mod =  (a_pow % N as i64) as i32;
        println!("a = {}, pow = {}, mod = {}", a, pow, N);
        println!("a_pow_mod = {}", a_pow_mod);

        let func = |x: i32| -> i32 {
            (x * a_pow_mod) % N as i32
        };
        let u_gate = operation::to_qunatum_gate(&func, targets.clone());
        let c_u_gate = operation::to_controlled(u_gate, i);
        println!("applying c_u_gate for i = {}", i);
        reg.apply(&c_u_gate);
        println!("Applied c_u_gate for i = {}", i);
    }
    // let qft = operation::qft(2_i32.pow(2*n as u32) as usize); 
    let qft = operation::qft(2_i32.pow(2*n as u32) as usize); 
    // reg.print_nonzero_probabilities();
    println!("applying qft");
    reg.apply(&qft);
    println!("applied qft");
    
    // measure the first 2n qubits
    let mut res =0;

    // reg.print_nonzero_probabilities();
    println!("measuring");
    for i in 0..2*n {
        let m = if reg.measure(i) {1} else {0};
        res |= m<<i;
    }
    println!("measured");
    // println!("res = {:?}", res);
    // reg.print_nonzero_probabilities();
    let dem = 2_i32.pow(2*n as u32) as u32;
    let ret =  dem / gcd::euclid_u32(res, dem);
    println!("ret = {}", ret);
    ret

}

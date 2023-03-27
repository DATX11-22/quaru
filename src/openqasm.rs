use crate::{operation, register::Register};
use openqasm_parser::openqasm::{self, BasicOp};
use std::{collections::HashMap, path::Path};

pub struct Registers {
    pub qregs: HashMap<String, Register>,
    pub cregs: HashMap<String, Vec<bool>>,
}

pub fn run_openqasm(openqasm_file: &Path) -> Registers {
    let program = openqasm::parse_openqasm(openqasm_file).expect("Could not parse openqasm file");

    let mut registers = Registers {
        qregs: program
            .qregs
            .iter()
            .map(|(name, &size)| (name.clone(), Register::new(&vec![false; size])))
            .collect(),
        cregs: program
            .cregs
            .iter()
            .map(|(name, &size)| (name.clone(), vec![false; size]))
            .collect(),
    };

    for (condition, op) in program.get_basic_operations() {
        if let Some(condition) = condition {
            let creg = registers
                .cregs
                .get(&condition.0)
                .expect("Register does not exist?");

            if creg_to_u32(creg) != condition.1 {
                continue;
            }
        }

        match op {
            BasicOp::U(p1, p2, p3, q) => {
                let qreg = registers.qregs.get_mut(&q.0).expect("Register does not exist?");
                qreg.apply(&operation::u(p1 as f64, p2 as f64, p3 as f64, q.1));
            }
            BasicOp::CX(q1, q2) => {
                assert_eq!(q1.0, q2.0, "The simulator currently does not support applying CX on qubits in two different registers");
                let qreg = registers
                    .qregs
                    .get_mut(&q1.0)
                    .expect("Register does not exist?");
                qreg.apply(&operation::cnot(q1.1, q2.1));
            }
            BasicOp::Measure(q, c) => {
                let qreg = registers.qregs.get_mut(&q.0).expect("Register does not exist?");
                let creg = registers.cregs.get_mut(&c.0).expect("Register does not exist?");

                creg[c.1] = qreg.measure(q.1);
            }
            BasicOp::ResetQ(q) => {
                let qreg = registers.qregs.get_mut(&q.0).expect("Register does not exist?");
                *qreg = Register::new(&vec![false; qreg.size()]);
            }
            BasicOp::ResetC(c) => {
                let creg = registers.cregs.get_mut(&c.0).expect("Register does not exist?");
                *creg = vec![false; creg.len()];
            }
        };
    }

    registers
}

fn creg_to_u32(creg: &Vec<bool>) -> u32 {
    let mut res = 0;

    for bit in creg.iter().rev() {
        res = res << 1;
        if *bit {
            res |= 1;
        }
    }

    res
}

use crate::cli_utils::display::IdentfiableOperation;
use crate::register::Register;

use std::fmt::Display;
use std::collections::HashMap;

/// Initial choices
pub enum Choice {
    Show,
    Circuit,
    Apply,
    Measure,
    Create,
    OpenQASM,
    Exit,
}

impl Choice {
    /// The possible actions the user can take. Which actions are available depend on the state
    /// of the simulation.
    pub fn choices(state: &State) -> Vec<Choice> {
        let mut choices = Vec::new();
        if !state.q_regs.is_empty() || !state.c_regs.is_empty() {
            choices.append(&mut vec![Choice::Show]);
        }
        if !state.q_regs.is_empty() {
            choices.append(&mut vec![Choice::Apply, Choice::Measure, Choice::Circuit]);
        }

        choices.append(&mut vec![Choice::Create, Choice::OpenQASM, Choice::Exit]);

        choices
    }
}

impl Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Choice::Apply => write!(f, "Apply"),
            Choice::Measure => write!(f, "Measure"),
            Choice::Show => write!(f, "Show"),
            Choice::Circuit => write!(f, "Show Circuit"),
            Choice::Create => write!(f, "Create register"),
            Choice::OpenQASM => write!(f, "Run OpenQASM program"),
            Choice::Exit => write!(f, "Exit"),
        }
    }
}

#[derive(Debug)]
pub enum OperationType {
    Unary,
    Binary,
}

impl OperationType {
    pub fn types() -> Vec<OperationType> {
        vec![OperationType::Unary, OperationType::Binary]
    }

    pub fn size(&self) -> usize {
        match self {
            OperationType::Unary => 1,
            OperationType::Binary => 2,
        }
    }
}

impl Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::Unary => write!(f, "Unary"),
            OperationType::Binary => write!(f, "Binary"),
        }
    }
}

pub enum UnaryOperation {
    Identity,
    Hadamard,
    Phase,
    Not,
    PauliY,
    PauliZ,
}

impl UnaryOperation {
    /// Returns a vector of every possible unary operation.
    pub fn operations() -> Vec<UnaryOperation> {
        vec![
            UnaryOperation::Identity,
            UnaryOperation::Hadamard,
            UnaryOperation::Phase,
            UnaryOperation::Not,
            UnaryOperation::PauliY,
            UnaryOperation::PauliZ,
        ]
    }

    pub fn target_name(_: &UnaryOperation) -> [&str; 1] {
        ["target"]
    }
}

impl Display for UnaryOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOperation::Identity => write!(f, "Identity"),
            UnaryOperation::Hadamard => write!(f, "Hadamard"),
            UnaryOperation::Phase => write!(f, "Phase"),
            UnaryOperation::Not => write!(f, "NOT"),
            UnaryOperation::PauliY => write!(f, "Pauli Y"),
            UnaryOperation::PauliZ => write!(f, "Pauli Z"),
        }
    }
}


pub enum BinaryOperation {
    CNot,
    Swap,
}

impl BinaryOperation {
    /// Returns a vector of every possible binary operation.
    pub fn operations() -> Vec<BinaryOperation> {
        vec![BinaryOperation::CNot, BinaryOperation::Swap]
    }
    
    pub fn target_names(op: &BinaryOperation) -> [&str; 2] {
        match *op {
            BinaryOperation::CNot => ["control", "target"],
            _ => ["target"; 2],
        }
    }
}

impl Display for BinaryOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperation::CNot => write!(f, "CNOT"),
            BinaryOperation::Swap => write!(f, "Swap"),
        }
    }
}


#[derive(Debug)]
pub enum RegisterType {
    Classical,
    Quantum,
}

impl Display for RegisterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

pub type RegCollection<T> = HashMap<String, T>;
pub type QRegCollection = RegCollection<HistoryQRegister>;
pub type CRegCollection = RegCollection<Vec<bool>>;

/// The state of the simulator
pub struct State {
    q_regs: QRegCollection,
    c_regs: CRegCollection,
}

impl State {
    pub fn new() -> State {
        Self { q_regs: HashMap::new(), c_regs: HashMap::new() }
    }
    pub fn q_regs(&self) -> QRegCollection {
        self.q_regs.clone()
    }
    pub fn c_regs(&self) -> CRegCollection {
        self.c_regs.clone()
    }

    pub fn q_regs_as_mut(&mut self) -> &mut QRegCollection {
        &mut self.q_regs
    }

    pub fn c_regs_as_mut(&mut self) -> &mut CRegCollection {
        &mut self.c_regs
    }
    
    pub fn insert_q_reg(&mut self, key: String, val: HistoryQRegister) {
        self.q_regs.insert(key, val);
    }

    pub fn insert_c_reg(&mut self, key: String, val: Vec<bool>) {
        self.c_regs.insert(key, val);
    }

    pub fn extend_c_regs(&mut self, other: CRegCollection) {
        self.c_regs.extend(other)
    }

    pub fn extend_q_regs(&mut self, other: QRegCollection) {
        self.q_regs.extend(other)
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct HistoryQRegister {
    register: Register,
    history: Vec<IdentfiableOperation>,
}

impl HistoryQRegister  {
    pub fn new(input_bits: &[bool]) -> HistoryQRegister {
        let register = Register::new(input_bits);
        let history: Vec<IdentfiableOperation> = Vec::new();

        HistoryQRegister { register, history }
    }

    pub fn size(&self) -> usize {
        self.register.size()
    }

    pub fn history(&self) -> Vec<IdentfiableOperation> {
        self.history.clone()
    }

    pub fn print_state(&self) {
        self.register.print_state()
    }

    pub fn apply(&mut self, op: &IdentfiableOperation) {
        self.history.push(op.clone());
        self.register.apply(&op.operation());
    }

    pub fn from_register(register: Register) -> HistoryQRegister {
        HistoryQRegister { register, history: Vec::new() }
    }

    pub fn measure(&mut self, index: usize) -> bool {
        self.register.measure(index)
    }
}



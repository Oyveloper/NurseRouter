use std::collections::HashMap;

use super::patient::Patient;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Depot {
    pub return_time: f32,
    pub x_coord: f32,
    pub y_coord: f32,
}

pub fn load_problem_instance(path: &str) -> ProblemInstance {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let problem_instance: ProblemInstance = serde_json::from_reader(reader).unwrap();
    problem_instance
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProblemInstance {
    pub nbr_nurses: u32,
    pub benchmark: f64,
    pub capacity_nurse: u32,
    pub depot: Depot,
    pub patients: HashMap<String, Patient>,
    pub travel_times: Vec<Vec<f32>>,
    pub instance_name: String,
}

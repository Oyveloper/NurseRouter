use rand::seq::SliceRandom;

use super::problem_instance::ProblemInstance;
use std::{cell::Ref, fmt};

#[derive(Clone)]
pub struct Solution {
    pub routes: Vec<Vec<String>>,
}

#[derive(Debug)]
pub struct SolutionEvaluation {
    pub total_travel_time: f32,
    pub number_of_time_window_violations: u32,
    pub number_of_return_time_violations: u32,
    pub sum_capacity_violation: u32,
}

impl Solution {
    pub fn random(problem_instance: Ref<ProblemInstance>) -> Solution {
        let mut patients: Vec<String> = problem_instance.patients.keys().cloned().collect();
        patients.shuffle(&mut rand::thread_rng());
        let mut routes = Vec::new();

        for _ in 0..problem_instance.nbr_nurses {
            let route = Vec::new();
            routes.push(route);
        }

        let mut i = 0;
        while patients.len() > 0 {
            routes[i].push(patients.pop().unwrap());
            i += 1;
            if i >= problem_instance.nbr_nurses as usize {
                i = 0;
            }
        }

        Solution { routes }
    }

    pub fn evaluation(&self, problem_instance: Ref<ProblemInstance>) -> SolutionEvaluation {
        let mut total_travel_time = 0.0;
        let mut number_of_time_window_violations = 0;
        let mut number_of_return_time_violations = 0;
        let mut sum_capacity_violation = 0;
        for route in &self.routes {
            // Initialize time
            let mut t = 0.0;
            let mut prev_patient_index = 0 as usize;
            let mut demand = 0.0;

            for patient_id in route {
                //Travel to next stop
                let patient_index = patient_id.parse::<u32>().unwrap() as usize;
                let travel_time = problem_instance.travel_times[prev_patient_index][patient_index];
                total_travel_time += travel_time;

                // Increment time
                t += travel_time;

                // Check patient window
                let patient = &problem_instance.patients[patient_id];
                if t < patient.start_time {
                    // Wait for window to start
                    t = patient.start_time;
                } else if t > patient.end_time {
                    number_of_time_window_violations += 1;
                }

                // Perform the service
                t += patient.care_time;
                demand += patient.demand;

                if t > patient.end_time {
                    number_of_time_window_violations += 1;
                }

                prev_patient_index = patient_index;
            }

            // Also calculate time to get back to depot
            let travel_time = problem_instance.travel_times[prev_patient_index][0];
            total_travel_time += travel_time;
            t += travel_time;

            // Check return violation
            if t > problem_instance.depot.return_time {
                number_of_return_time_violations += 1;
            }

            // Check demand violation
            if demand > problem_instance.capacity_nurse as f32 {
                sum_capacity_violation += 1;
            }
        }

        SolutionEvaluation {
            total_travel_time,
            number_of_time_window_violations,
            number_of_return_time_violations,
            sum_capacity_violation,
        }
    }
}

impl fmt::Debug for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}",
            serde_json::to_string(
                &self
                    .routes
                    .iter()
                    .map(|r| r
                        .iter()
                        .map(|s| s.parse::<u32>().or::<u32>(Ok(0)).unwrap())
                        .collect::<Vec<u32>>())
                    .collect::<Vec<Vec<u32>>>()
            )
            .unwrap()
        )
    }
}

pub fn is_solution_valid(solution: &Solution, problem_instance: Ref<ProblemInstance>) -> bool {
    if solution.routes.len() != problem_instance.nbr_nurses as usize {
        return false;
    }

    let all_patients_in_solution = solution.routes.iter().flatten().collect::<Vec<_>>();
    let all_patients_in_problem_instance = problem_instance.patients.keys().collect::<Vec<_>>();

    if all_patients_in_solution != all_patients_in_problem_instance {
        return false;
    }

    let evaluation = solution.evaluation(problem_instance);

    if evaluation.number_of_time_window_violations > 0 {
        return false;
    }
    if evaluation.number_of_return_time_violations > 0 {
        return false;
    }

    if evaluation.sum_capacity_violation > 0 {
        return false;
    }

    true
}

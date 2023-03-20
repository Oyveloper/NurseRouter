use rand::{seq::SliceRandom, Rng};
use serde::Serialize;

use super::problem_instance::ProblemInstance;
use std::fmt;

#[derive(Clone, Serialize, Default)]
pub struct Solution {
    pub force_valid: bool,
    pub routes: Vec<Vec<String>>,
    pub fitness: Option<SolutionFitness>,
}

#[derive(Clone, Copy, Serialize, Default)]
pub struct SolutionFitness {
    pub penalized: f64,
    pub unpenalized: f64,
    pub score: f64,
}

#[derive(Debug)]
pub struct SolutionEvaluation {
    pub total_travel_time: f64,
    pub number_of_time_window_violations: u32,
    pub number_of_return_time_violations: u32,
    pub sum_capacity_violation: u32,
    pub route_info: Vec<RouteInfo>,
}

#[derive(Debug)]
pub struct RouteInfo {
    pub total_travel_time: f32,
    pub nurse_visits: Vec<NurseVisit>,
    pub covered_demand: f32,
    pub return_time: f32,
}

#[derive(Debug)]
pub struct NurseVisit {
    pub patient_id: String,
    pub start_time: f32,
    pub end_time: f32,
    pub window_start: f32,
    pub window_end: f32,
}

impl Solution {
    pub fn random(problem_instance: &ProblemInstance) -> Solution {
        let mut rng = rand::thread_rng();
        let mut patients: Vec<String> = problem_instance.patients.keys().cloned().collect();
        patients.shuffle(&mut rng);
        // patients.sort_by_key(|patient_id| {
        //     let patient = &problem_instance.patients[patient_id];
        //     patient.end_time.round() as u32
        // });
        let mut routes = Vec::new();

        for _ in 0..problem_instance.nbr_nurses {
            let route = Vec::new();
            routes.push(route);
        }

        let mut i = 0;
        while patients.len() > 0 {
            let nbr_patients = if patients.len() == 1 {
                1
            } else {
                rng.gen_range(0..patients.len())
            };
            for _ in 0..nbr_patients {
                routes[i].push(patients.pop().unwrap());
            }

            i += 1;
            if i >= problem_instance.nbr_nurses as usize {
                i = 0;
            }
        }

        // for route in &mut routes {
        //     // route.shuffle(&mut rand::thread_rng());
        //     // route.sort_by_key(|patient_id| {
        //     //     let patient = &problem_instance.patients[patient_id];
        //     //     patient.end_time.round() as u32
        //     // });
        // }

        Solution {
            routes,
            force_valid: false,
            ..Default::default()
        }
    }

    pub fn k_means(problem_instance: &ProblemInstance) -> Solution {
        let max_iter = 100;
        let patients: Vec<i32> = problem_instance
            .patients
            .keys()
            .cloned()
            .map(|k| k.parse::<i32>().unwrap())
            .collect();

        let k = rand::thread_rng().gen_range(1..problem_instance.nbr_nurses as usize);

        let mut centers = (0..k)
            .map(|_| patients[rand::thread_rng().gen_range(0..patients.len())].clone())
            .collect::<Vec<i32>>();

        let mut clusters = vec![Vec::new(); problem_instance.nbr_nurses as usize];

        for _ in 0..max_iter {
            for cluster in clusters.iter_mut() {
                cluster.clear();
            }
            for patient in patients.iter() {
                let mut min_dist = std::f32::MAX;
                let mut min_index = 0;
                for (i, center) in centers.iter().enumerate() {
                    let dist = problem_instance.travel_times[*patient as usize][*center as usize];
                    if dist < min_dist {
                        min_dist = dist;
                        min_index = i;
                    }
                }
                clusters[min_index].push(patient);
            }

            let mut new_centers = Vec::new();
            for cluster in clusters.iter() {
                let mut min_dist = std::f32::MAX;
                let mut min_index = 0;
                for (i, patient) in cluster.iter().enumerate() {
                    let mut dist = 0.0;
                    for other_patient in cluster.iter() {
                        dist += problem_instance.travel_times[**patient as usize]
                            [**other_patient as usize];
                    }
                    if dist < min_dist {
                        min_dist = dist;
                        min_index = i;
                    }
                }
                if cluster.len() == 0 {
                    new_centers.push(0);
                    continue;
                }
                new_centers.push(*cluster[min_index]);
            }

            if new_centers
                .iter()
                .zip(centers.iter())
                .any(|(a, b)| *a != *b)
            {
                centers = new_centers;
            } else {
                break;
            }
        }

        // for route in &mut routes {
        //     // route.shuffle(&mut rand::thread_rng());
        //     // route.sort_by_key(|patient_id| {
        //     //     let patient = &problem_instance.patients[patient_id];
        //     //     patient.end_time.round() as u32
        //     // });
        // }

        let routes: Vec<Vec<String>> = clusters
            .iter()
            .map(|c| {
                let mut c = c.clone();
                c.sort_by_key(|patient_id| {
                    let patient = &problem_instance.patients[&patient_id.to_string()];
                    patient.end_time.round() as u32
                });
                c.iter().map(|i| i.to_string()).collect()
            })
            .collect();

        Solution {
            routes,
            force_valid: true,
            ..Default::default()
        }
    }

    pub fn evaluation(&self, problem_instance: &ProblemInstance) -> SolutionEvaluation {
        let mut total_travel_time = 0.0;
        let mut number_of_time_window_violations = 0;
        let mut number_of_return_time_violations = 0;
        let mut sum_capacity_violation = 0;
        let mut route_infos = Vec::new();

        for route in &self.routes {
            // Initialize time
            let mut t = 0.0;
            let mut prev_patient_index = 0 as usize;
            let mut demand = 0.0;
            let mut nurse_visits = Vec::new();

            for patient_id in route {
                //Travel to next stop
                let patient_index = patient_id.parse::<u32>().unwrap() as usize;
                let travel_time = problem_instance.travel_times[prev_patient_index][patient_index];
                total_travel_time += travel_time as f64;

                // Increment time
                t += travel_time;

                // Check patient window
                let patient = &problem_instance.patients[patient_id];
                if t < patient.start_time {
                    // Wait for window to start
                    t = patient.start_time;
                }

                // Perform the service
                t += patient.care_time;
                demand += patient.demand;

                if t > patient.end_time {
                    number_of_time_window_violations += 1;
                }

                prev_patient_index = patient_index;
                nurse_visits.push(NurseVisit {
                    patient_id: patient_id.clone(),
                    start_time: t - patient.care_time,
                    end_time: t,
                    window_start: patient.start_time,
                    window_end: patient.end_time,
                });
            }

            // Also calculate time to get back to depot
            let travel_time = problem_instance.travel_times[prev_patient_index][0];
            total_travel_time += travel_time as f64;
            t += travel_time;

            // Check return violation
            if t > problem_instance.depot.return_time {
                number_of_return_time_violations += 1;
            }

            // Check demand violation
            if demand > problem_instance.capacity_nurse as f32 {
                sum_capacity_violation += 1;
            }

            route_infos.push(RouteInfo {
                total_travel_time: travel_time,
                nurse_visits,
                covered_demand: demand,
                return_time: t,
            });
        }

        SolutionEvaluation {
            total_travel_time,
            number_of_time_window_violations,
            number_of_return_time_violations,
            sum_capacity_violation,
            route_info: route_infos,
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

pub fn _find_violation_indices(
    solution: &Solution,
    problem_instance: &ProblemInstance,
) -> Vec<(usize, usize)> {
    let mut res = Vec::new();

    for (i, route) in solution.routes.iter().enumerate() {
        let mut t = 0.0;

        let mut prev_patient_index = 0;
        for (j, patient_id) in route.iter().enumerate() {
            let mut time_violated = false;
            //Travel to next stop
            let patient_index = patient_id.parse::<u32>().unwrap() as usize;
            let travel_time = problem_instance.travel_times[prev_patient_index][patient_index];

            // Increment time
            t += travel_time;

            // Check patient window
            let patient = &problem_instance.patients[patient_id];
            if t < patient.start_time {
                // Wait for window to start
                t = patient.start_time;
            } else if t > patient.end_time {
                time_violated = true;
            }

            // Perform the service
            t += patient.care_time;

            if t > patient.end_time {
                time_violated = true;
            }

            if time_violated {
                res.push((i, j));
            }

            prev_patient_index = patient_index;
        }
    }

    res
}

pub fn is_solution_valid(solution: &Solution, problem_instance: &ProblemInstance) -> bool {
    if solution.routes.len() != problem_instance.nbr_nurses as usize {
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

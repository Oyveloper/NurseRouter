use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
};

use crossbeam_channel::unbounded;
use indicatif::ProgressBar;
use rand::Rng;
use serde::Serialize;

use crate::{
    ga::ga::NurseRouteGA,
    model::{
        problem_instance::{load_problem_instance, ProblemInstance},
        solution::{Solution, SolutionFitness},
    },
};

pub struct SolverResult(pub Solution, pub Vec<(Vec<f64>, Vec<f64>)>);

#[derive(Serialize, Clone)]
pub struct SolverConfig {
    pub invalid_fraction: f64,
    pub unpenalized_population_fraction: f64,
    pub population_size: u32,
    pub generation_gap: u32,
    pub nbr_generations: i32,
    pub nbr_threads: usize,
    pub nbr_epochs: usize,
    pub nbr_migrants: u32,
    pub mutation_probability: f64,
    pub crossover_probability: f64,
    pub lin_rank_preassure: f32,
    pub violation_penalty: f64,
    pub random_offspring_variation_limit: f64,
    pub random_offspring_variation_window_size: i32,
    pub large_neighborhood_improve_probability: f64,
    pub local_search_probability: f64,
    pub local_search_iters: usize,
    pub problem_instance_location: String,

    #[serde(skip)]
    pub eval: fn(&Solution, &ProblemInstance, f64) -> SolutionFitness,
}

pub fn solve_problem(config: SolverConfig) -> SolverResult {
    let problem_instance = load_problem_instance(config.problem_instance_location.clone().as_str());

    // Setting up thread communication
    let (history_sender, history_receiver) = mpsc::channel();
    let (best_sender, best_receiver) = mpsc::channel();
    let mut channels = Vec::new();
    for _ in 0..config.nbr_threads {
        let (sender, receiver) = unbounded::<Vec<Solution>>();
        channels.push((sender, receiver));
    }
    let (progress_sender, progress_receiver) = mpsc::channel();
    let mut threads = Vec::new();
    let nbr_unpenalized_threads = (config.invalid_fraction * config.nbr_threads as f64) as usize;

    static SHUTDOWN: AtomicBool = AtomicBool::new(false);

    for i in 0..config.nbr_threads {
        let (_, receiver) = channels[i].clone();

        let senders = channels
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, (s, _))| s.clone())
            .collect::<Vec<_>>();

        let history_sender_clone = history_sender.clone();
        let best_sender_clone = best_sender.clone();
        let progress_sender_clone = progress_sender.clone();

        let mut violation_penalty = config.violation_penalty;
        let config_location = config.problem_instance_location.clone();

        let t = thread::spawn(move || {
            // Load problem instance for each thread
            // TODO: make it possible to share this
            let problem_instance_t = load_problem_instance(config_location.as_str());
            let mut mutation_probability = config.mutation_probability;
            let mut crossover_probability = 0.0;
            let mut lin_rank_preassure = config.lin_rank_preassure;

            if i < nbr_unpenalized_threads {
                violation_penalty = 0.0;
                mutation_probability = 0.2;
                crossover_probability = 0.1;
                lin_rank_preassure = 1.1;
            }

            let mut algo = NurseRouteGA::new(
                config.population_size,
                config.generation_gap,
                config.unpenalized_population_fraction,
                lin_rank_preassure,
                config.eval,
                violation_penalty,
                &problem_instance_t,
                mutation_probability,
                crossover_probability,
                config.random_offspring_variation_limit,
                config.random_offspring_variation_window_size,
                config.large_neighborhood_improve_probability,
                config.local_search_probability,
                config.local_search_iters,
                progress_sender_clone.clone(),
            );

            for i in 0..config.nbr_epochs {
                algo.run(config.nbr_generations);

                let best_solution = algo.get_best_solution();
                let score = (config.eval)(&best_solution, &problem_instance_t, violation_penalty)
                    .unpenalized;

                if (score - problem_instance.benchmark).abs() < 0.05 {
                    SHUTDOWN.store(true, Ordering::Relaxed);
                }

                if SHUTDOWN.load(Ordering::Relaxed) {
                    break;
                }

                if i >= 1 {
                    //config.nbr_epochs /  {
                    algo.crossover_probability = config.crossover_probability;
                }

                for sender in senders.iter() {
                    sender.send(algo._emigrate(config.nbr_migrants)).unwrap();
                }

                for _ in 0..senders.len() {
                    if rand::thread_rng().gen_bool(0.2) {
                        let im = receiver.recv().unwrap();
                        algo._immigrate(im);
                    }
                }
            }

            history_sender_clone
                .send(algo.get_fitness_history())
                .unwrap();

            let mut best_solution = algo.get_best_solution().clone();

            for _ in 0..200 {
                best_solution = algo.do_local_search(best_solution);
            }

            best_sender_clone.send(best_solution).unwrap();
        });

        threads.push(t);
    }

    let mut progress_count = 0;
    let target_progress_count =
        config.nbr_threads * config.nbr_epochs * config.nbr_generations as usize;

    let progress = ProgressBar::new(target_progress_count as u64);
    while progress_count < target_progress_count {
        progress_receiver.recv().unwrap();
        progress.inc(1);
        progress_count += 1;
    }

    for t in threads {
        t.join().unwrap();
    }

    let mut best_solutions = Vec::new();
    let mut histories = Vec::new();

    for _ in 0..config.nbr_threads {
        let history = history_receiver.recv().unwrap();
        histories.push(history);

        let best_solution = best_receiver.recv().unwrap();
        best_solutions.push(best_solution);
    }

    // let mut final_ga = NurseRouteGA::new(
    //     config.population_size,
    //     config.generation_gap,
    //     0.1,
    //     config.lin_rank_preassure,
    //     config.eval,
    //     config.violation_penalty,
    //     &problem_instance,
    //     config.mutation_probability,
    //     config.crossover_probability,
    //     config.random_offspring_variation_limit,
    //     config.random_offspring_variation_window_size,
    //     config.large_neighborhood_improve_probability,
    //     config.local_search_probability,
    //     config.local_search_iters,
    //     progress_sender.clone(),
    // );
    //
    // final_ga.population.append(&mut best_solutions);
    // final_ga.run(config.nbr_generations);
    //
    // let best_solution = final_ga.get_best_solution();

    let best_solution = best_solutions
        .iter()
        .max_by_key(|s| (config.eval)(s, &problem_instance, 1000.0).penalized as i32)
        .unwrap();

    SolverResult(best_solution.clone(), histories)
}

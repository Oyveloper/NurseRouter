mod ga;
mod model;
mod util;
use crossbeam_channel::unbounded;
use indicatif::ProgressBar;
use std::cell::{Ref, RefCell};
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use ga::ga::NurseRouteGA;
use model::problem_instance::{load_problem_instance, ProblemInstance};
use model::solution::Solution;

use crate::model::solution::is_solution_valid;
use plotters::prelude::*;

const PENALTY: f32 = 1000.0;
const MUTATION_PROBABILITY: f64 = 0.003;
const CROSSOVER_PROBABILITY: f64 = 0.9;
const POPULATION_SIZE: u32 = 150;
const OFFSPRING_SIZE: u32 = 150;
const LIN_RANK_PREASSURE: f32 = 0.8;
const NUMBER_OF_GENERATIONS: i32 = 50;

fn eval(solution: &Solution, problem_instance: Ref<ProblemInstance>) -> f32 {
    let evaluation = solution.evaluation(problem_instance);

    return -(evaluation.total_travel_time
        + PENALTY
            * (evaluation.number_of_time_window_violations as f32
                + evaluation.number_of_return_time_violations as f32
                + evaluation.sum_capacity_violation as f32))
        .powi(2);
}

fn travel_time(solution: &Solution, problem_instance: Ref<ProblemInstance>) -> f32 {
    let evaluation = solution.evaluation(problem_instance);

    return evaluation.total_travel_time;
}

fn main() {
    let problem_instance = Rc::new(RefCell::new(load_problem_instance(
        "assets/instances/train_9.json",
    )));

    let nbr_threads = 8;
    let nbr_epochs = 30;

    let (history_sender, history_receiver) = mpsc::channel();
    let (best_sender, best_receiver) = mpsc::channel();

    let mut channels = Vec::new();

    for _ in 0..nbr_threads {
        let (sender, receiver) = unbounded::<Vec<Solution>>();
        channels.push((sender, receiver));
    }

    let (progress_sender, progress_receiver) = mpsc::channel();

    let mut threads = Vec::new();

    for i in 0..nbr_threads {
        let (_, receiver) = channels[i].clone();
        let next_i = if i == nbr_threads - 1 { 0 } else { i + 1 };
        let (sender, _) = channels[next_i].clone();
        let history_sender_clone = history_sender.clone();
        let best_sender_clone = best_sender.clone();
        let progress_sender_clone = progress_sender.clone();

        // {
        //     let mut th = training_history.lock().unwrap();
        //     th.push(Vec::new());
        // }
        let t = thread::spawn(move || {
            // Load problem instance for each thread
            // TODO: make it possible to share this
            let problem_instance_t = Rc::new(RefCell::new(load_problem_instance(
                "assets/instances/train_0.json",
            )));

            let mut algo = NurseRouteGA::new(
                POPULATION_SIZE,
                OFFSPRING_SIZE,
                LIN_RANK_PREASSURE,
                eval,
                problem_instance_t,
                MUTATION_PROBABILITY,
                CROSSOVER_PROBABILITY,
            );

            for _ in 0..nbr_epochs {
                algo.run(NUMBER_OF_GENERATIONS);
                progress_sender_clone.send(()).unwrap();

                sender.send(algo.emigrate(10)).unwrap();
                let imigrants = receiver.recv();
                match imigrants {
                    Ok(ref imigrants) => {
                        algo.immigrate(imigrants.clone());
                    }
                    Err(_) => {}
                }
            }

            history_sender_clone
                .send(algo.get_fitness_history())
                .unwrap();

            best_sender_clone.send(algo.get_best_solution()).unwrap();
        });

        threads.push(t);
    }

    let mut progress_count = 0;
    let target_progress_count = nbr_threads * nbr_epochs;

    let progress = ProgressBar::new(target_progress_count as u64);
    while progress_count < target_progress_count {
        progress_receiver.recv().unwrap();
        progress.inc(1);
        progress_count += 1;
    }

    for t in threads {
        t.join().unwrap();
    }

    // let mut algo = NurseRouteGA::new(
    //     POPULATION_SIZE,
    //     OFFSPRING_SIZE,
    //     LIN_RANK_PREASSURE,
    //     eval,
    //     problem_instance.clone(),
    //     MUTATION_PROBABILITY,
    //     CROSSOVER_PROBABILITY,
    // );
    // let best_solution = algo.get_best_solution();
    // algo.run(NUMBER_OF_GENERATIONS);
    // let (best, avg) = algo.get_fitness_history();
    //
    // let new_best_solution = algo.get_best_solution();
    //
    // println!("{:#?}", best_solution);
    // println!("{:#?}", new_best_solution);
    //
    // println!(
    //     "{:#?}",
    //     travel_time(&best_solution, problem_instance.borrow())
    // );
    // println!(
    //     "{:#?}",
    //     travel_time(&new_best_solution, problem_instance.borrow())
    // );
    //
    // println!(
    //     "{:#?}",
    //     is_solution_valid(&best_solution, problem_instance.borrow())
    // );
    //
    // println!(
    //     "{:#?}",
    //     new_best_solution.evaluation(problem_instance.borrow())
    // );
    //
    let root_area = BitMapBackend::new("images/GA_plot.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Fitness", ("sans-serif", 40))
        .build_cartesian_2d(0.0..((NUMBER_OF_GENERATIONS * 5) as f64), 0.0..260000.0)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    let mut best_solutions = Vec::new();

    for _ in 0..nbr_threads {
        let history = history_receiver.recv().unwrap();
        let (best, avg) = history;

        let best_solution = best_receiver.recv().unwrap();
        best_solutions.push(best_solution);

        ctx.draw_series(LineSeries::new(
            best.iter()
                .enumerate()
                .map(|(i, v)| (i as f64, -v.clone() as f64 / 10000.0)),
            &BLUE,
        ))
        .unwrap();

        ctx.draw_series(LineSeries::new(
            avg.iter()
                .enumerate()
                .map(|(i, v)| (i as f64, -v.clone() as f64 / 10000.0)),
            &RED,
        ))
        .unwrap();
    }

    let best_solution = best_solutions
        .iter()
        .min_by_key(|s| eval(s, problem_instance.borrow()) as i32)
        .unwrap();

    println!("{:#?}", best_solution);
    println!("{:#?}", best_solution.evaluation(problem_instance.borrow()));
    println!(
        "{:#?}",
        is_solution_valid(best_solution, problem_instance.borrow())
    );
}

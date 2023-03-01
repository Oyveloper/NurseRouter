mod ga;
mod model;
mod util;
use crossbeam_channel::unbounded;
use indicatif::ProgressBar;
use std::cell::{Ref, RefCell};
use std::env::args;
use std::rc::Rc;
use std::sync::mpsc;
use std::thread;

use ga::ga::NurseRouteGA;
use model::problem_instance::{load_problem_instance, ProblemInstance};
use model::solution::Solution;

use crate::model::solution::is_solution_valid;
use plotters::prelude::*;

const PENALTY: f64 = 4000.0;
const MUTATION_PROBABILITY: f64 = 0.00005;
const CROSSOVER_PROBABILITY: f64 = 0.998;
const POPULATION_SIZE: u32 = 200;
const OFFSPRING_SIZE: u32 = 200;
const LIN_RANK_PREASSURE: f32 = 1.998;
const NUMBER_OF_GENERATIONS: i32 = 25;
const NBR_THREADS: usize = 25;
const NBR_EPOCHS: usize = 40;
const NBR_MIGRANTS: u32 = 2;

fn eval(solution: &Solution, problem_instance: Ref<ProblemInstance>) -> f64 {
    let evaluation = solution.evaluation(problem_instance);

    -(evaluation.total_travel_time
        + PENALTY
            * (evaluation.number_of_time_window_violations as f64
                + evaluation.number_of_return_time_violations as f64
                + evaluation.sum_capacity_violation as f64))
}

fn _travel_time(solution: &Solution, problem_instance: Ref<ProblemInstance>) -> f64 {
    let evaluation = solution.evaluation(problem_instance);

    return evaluation.total_travel_time;
}

fn main() {
    let problem_number = args()
        .nth(1)
        .unwrap_or_else(|| String::from("1"))
        .parse::<u32>()
        .unwrap();
    println!("Problem number: {}", problem_number);
    let problem_instance = Rc::new(RefCell::new(load_problem_instance(
        format!("assets/instances/train_{}.json", problem_number).as_str(),
    )));

    let (history_sender, history_receiver) = mpsc::channel();
    let (best_sender, best_receiver) = mpsc::channel();

    let mut channels = Vec::new();

    for _ in 0..NBR_THREADS {
        let (sender, receiver) = unbounded::<Vec<Solution>>();
        channels.push((sender, receiver));
    }

    let (progress_sender, progress_receiver) = mpsc::channel();

    let mut threads = Vec::new();

    for i in 0..NBR_THREADS {
        let (_, receiver) = channels[i].clone();
        let next_i = if i == NBR_THREADS - 1 { 0 } else { i + 1 };
        let (sender, _) = channels[next_i].clone();
        let history_sender_clone = history_sender.clone();
        let best_sender_clone = best_sender.clone();
        let progress_sender_clone = progress_sender.clone();

        let t = thread::spawn(move || {
            // Load problem instance for each thread
            // TODO: make it possible to share this
            let problem_instance_t = Rc::new(RefCell::new(load_problem_instance(
                format!("assets/instances/train_{}.json", problem_number).as_str(),
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

            for _ in 0..NBR_EPOCHS {
                algo.run(NUMBER_OF_GENERATIONS);
                progress_sender_clone.send(()).unwrap();

                sender.send(algo.emigrate(NBR_MIGRANTS)).unwrap();
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
    let target_progress_count = NBR_THREADS * NBR_EPOCHS;

    let progress = ProgressBar::new(target_progress_count as u64);
    while progress_count < target_progress_count {
        progress_receiver.recv().unwrap();
        progress.inc(1);
        progress_count += 1;
    }

    for t in threads {
        t.join().unwrap();
    }
    let root_area = BitMapBackend::new("images/GA_plot.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Fitness", ("sans-serif", 40))
        .build_cartesian_2d(
            0.0..((NUMBER_OF_GENERATIONS * NBR_EPOCHS as i32) as f64),
            0.0..5000.0,
        )
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    let mut best_solutions = Vec::new();

    for _ in 0..NBR_THREADS {
        let history = history_receiver.recv().unwrap();
        let (best, avg) = history;

        let best_solution = best_receiver.recv().unwrap();
        best_solutions.push(best_solution);

        ctx.draw_series(LineSeries::new(
            best.iter()
                .enumerate()
                .map(|(i, v)| (i as f64, -v.clone() as f64)),
            &BLUE,
        ))
        .unwrap();

        ctx.draw_series(LineSeries::new(
            avg.iter()
                .enumerate()
                .map(|(i, v)| (i as f64, -v.clone() as f64)),
            &RED,
        ))
        .unwrap();
    }

    let best_solution = best_solutions
        .iter()
        .max_by_key(|s| eval(s, problem_instance.borrow()) as i32)
        .unwrap();

    println!("{:#?}", best_solution);
    println!("{:#?}", best_solution.evaluation(problem_instance.borrow()));
    println!("{:#?}", eval(best_solution, problem_instance.borrow()));
    println!(
        "{:#?}",
        is_solution_valid(best_solution, problem_instance.borrow())
    );
}

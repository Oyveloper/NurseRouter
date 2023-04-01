mod ga;
mod model;
mod solver;
mod util;

use solver::solver::solve_problem;
use std::env::args;
use std::time::{SystemTime, UNIX_EPOCH};

use model::problem_instance::ProblemInstance;
use model::solution::{Solution, SolutionFitness};

use crate::model::problem_instance::load_problem_instance;
use crate::model::solution::is_solution_valid;
use crate::solver::solver::{SolverConfig, SolverResult};
use crate::util::output::{plot_histories, print_solution, save_solution_for_plotting};

const BASE_PENALTY: f64 = 500.0;
const INVALID_FRACTION: f64 = 0.0;
const UNPENALZED_POPULATION_FRACTION: f64 = 0.2;
const MUTATION_PROBABILITY: f64 = 0.05;
const CROSSOVER_PROBABILITY: f64 = 0.4;
const POPULATION_SIZE: u32 = 40;
const GENERATION_GAP: u32 = 10;
const LIN_RANK_PREASSURE: f32 = 1.90;
const NBR_GENERATIONS: i32 = 50;
const NBR_THREADS: usize = 10;
const NBR_EPOCHS: usize = 500;
const NBR_MIGRANTS: u32 = 5;

const LARGE_NEIGHBORHOOD_IMPROVE_PROBABILITY: f64 = 0.15;
const LOCAL_SEARCH_PROBABILITY: f64 = 0.5;
const LOCAL_SEARCH_ITERS: usize = 10;

const RANDOM_OFFSPRING_VARIATION_LIMIT: f64 = 0.01;
const RANDOM_OFFSPRING_VARIATION_WINDOW_SIZE: i32 = 500;

fn eval(
    solution: &Solution,
    problem_instance: &ProblemInstance,
    violation_penalty: f64,
) -> SolutionFitness {
    let evaluation = solution.evaluation(problem_instance);

    let unpenalized = -evaluation.total_travel_time;

    let penalty = violation_penalty
        * (evaluation.number_of_time_window_violations as f64
            + evaluation.number_of_return_time_violations as f64
            + evaluation.sum_capacity_violation as f64);

    let penalized = unpenalized - penalty;

    let score = if solution.force_valid {
        penalized
    } else {
        unpenalized
    };

    SolutionFitness {
        penalized,
        unpenalized,
        score,
    }
}

fn _travel_time(solution: &Solution, problem_instance: &ProblemInstance) -> f64 {
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

    let problem_path = format!("assets/instances/test_{}.json", problem_number);

    let config = SolverConfig {
        violation_penalty: BASE_PENALTY,
        invalid_fraction: INVALID_FRACTION,
        unpenalized_population_fraction: UNPENALZED_POPULATION_FRACTION,
        mutation_probability: MUTATION_PROBABILITY,
        crossover_probability: CROSSOVER_PROBABILITY,
        population_size: POPULATION_SIZE,
        generation_gap: GENERATION_GAP,
        lin_rank_preassure: LIN_RANK_PREASSURE,
        nbr_generations: NBR_GENERATIONS,
        nbr_epochs: NBR_EPOCHS,
        nbr_migrants: NBR_MIGRANTS,
        large_neighborhood_improve_probability: LARGE_NEIGHBORHOOD_IMPROVE_PROBABILITY,
        local_search_probability: LOCAL_SEARCH_PROBABILITY,
        local_search_iters: LOCAL_SEARCH_ITERS,
        random_offspring_variation_limit: RANDOM_OFFSPRING_VARIATION_LIMIT,
        random_offspring_variation_window_size: RANDOM_OFFSPRING_VARIATION_WINDOW_SIZE,
        nbr_threads: NBR_THREADS,
        problem_instance_location: problem_path.clone(),
        eval,
    };

    let problem_instance = load_problem_instance(problem_path.as_str());

    let SolverResult(best_solution, histories) = solve_problem(config.clone());

    let best_solution_score = eval(&best_solution, &problem_instance, 1000.0).penalized;

    print_solution(&best_solution, &problem_instance);
    // println!("{:#?}", best_solution.evaluation(&problem_instance));
    println!("{:#?}", best_solution_score);
    println!(
        "{:#?}",
        is_solution_valid(&best_solution, &problem_instance)
    );

    plot_histories(
        histories,
        (NBR_GENERATIONS * NBR_EPOCHS as i32) as f64,
        5000.0,
    );

    // prepare for plotting
    save_solution_for_plotting(
        &best_solution,
        best_solution_score,
        config,
        &format!(
            "images/plot_data/solution-{}-{:?}.json",
            problem_number,
            SystemTime::now().duration_since(UNIX_EPOCH)
        ),
    );
}

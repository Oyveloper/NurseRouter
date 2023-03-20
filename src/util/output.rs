use plotters::{prelude::*, style::WHITE};
use serde::Serialize;

use crate::{
    model::{problem_instance::ProblemInstance, solution::Solution},
    solver::solver::SolverConfig,
};

pub fn print_solution(solution: &Solution, problem_instance: &ProblemInstance) {
    println!("Nurse capacity: {}", problem_instance.capacity_nurse);
    println!("Depot return time: {}", problem_instance.depot.return_time);
    println!("----------------------------------------");
    let eval = solution.evaluation(problem_instance);

    for i in 0..solution.routes.len() {
        let route_info = &eval.route_info[i];
        let time = route_info.total_travel_time;
        let covered_demand = route_info.covered_demand;

        let mut route_strings = vec![String::from("D(0)")];

        for visit in route_info.nurse_visits.iter() {
            let visit_string = format!(
                "{}({}-{})[{}-{}]",
                visit.patient_id,
                visit.start_time,
                visit.end_time,
                visit.window_start,
                visit.window_end
            );
            route_strings.push(visit_string);
        }

        route_strings.push(format!("D({})", route_info.total_travel_time));

        let route_string = route_strings.join(" -> ");

        println!(
            "Nurse {}\t{}\t{}\t{}",
            i, time, covered_demand, route_string
        );

        println!();
    }

    println!("----------------------------------------");
    println!(
        "Objective value (total travel time): {}",
        eval.total_travel_time
    );
}

#[derive(Serialize)]
struct SolutionJsonSumary {
    routes: Vec<Vec<i32>>,
    score: f64,
    configuration: SolverConfig,
}

pub fn save_solution_for_plotting(
    solution: &Solution,
    score: f64,
    configuration: SolverConfig,
    output_path: &str,
) {
    let routes = &solution
        .routes
        .iter()
        .map(|r| {
            r.iter()
                .map(|v| v.parse::<i32>().unwrap())
                .collect::<Vec<i32>>()
        })
        .collect::<Vec<Vec<i32>>>();
    let output = SolutionJsonSumary {
        routes: routes.clone(),
        score,
        configuration,
    };
    std::fs::write(output_path, serde_json::to_string_pretty(&output).unwrap()).unwrap();
}

pub fn plot_histories(histories: Vec<(Vec<f64>, Vec<f64>)>, x_axis_len: f64, y_axis_len: f64) {
    let root_area = BitMapBackend::new("images/GA_plot.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Fitness", ("sans-serif", 40))
        .build_cartesian_2d(0.0..x_axis_len, 0.0..y_axis_len)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    for (best, avg) in histories {
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
}

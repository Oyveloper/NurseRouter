use indicatif::ProgressIterator;
use std::{
    cell::{Ref, RefCell},
    rc::Rc,
};

use crate::{
    model::{problem_instance::ProblemInstance, solution::Solution},
    util::sorting::argsort,
};
use rand::{
    distributions::{Standard, WeightedIndex},
    prelude::*,
    seq::SliceRandom,
};

pub enum CrossoverMode {
    Lecture,
}

pub enum MutationType {
    InRouteSwap,
    InRouteInvert,
    InRouteScramble,
    CrossRouteSwap,
}

impl Distribution<MutationType> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MutationType {
        match rng.gen_range(0..4) {
            0 => MutationType::InRouteSwap,
            1 => MutationType::InRouteInvert,
            2 => MutationType::InRouteScramble,
            3 => MutationType::CrossRouteSwap,
            _ => panic!("Invalid mutation type"),
        }
    }
}

pub struct NurseRouteGA {
    population: Vec<Solution>,
    population_size: u32,
    offspring_size: u32,
    lin_rank_pressure: f32,
    evaluation_fn: fn(&Solution, Ref<ProblemInstance>) -> f32,
    problem_instance: Rc<RefCell<ProblemInstance>>,
    population_evaluation: Vec<f32>,
    mutation_probability: f64,
    crossover_probability: f64,
    best_fitness_stats: Vec<f32>,
    avg_fitness_stats: Vec<f32>,
}

impl NurseRouteGA {
    pub fn new(
        population_size: u32,
        offspring_size: u32,
        lin_rank_pressure: f32,
        evaluation_fn: fn(&Solution, Ref<ProblemInstance>) -> f32,
        problem_instance: Rc<RefCell<ProblemInstance>>,
        mutation_probability: f64,
        crossover_probability: f64,
    ) -> NurseRouteGA {
        let best_fitness_stats = Vec::new();
        let avg_fitness_stats = Vec::new();
        NurseRouteGA {
            population: NurseRouteGA::generate_population(
                population_size,
                problem_instance.clone(),
            ),
            population_size,
            offspring_size,
            lin_rank_pressure,
            problem_instance: problem_instance.clone(),
            evaluation_fn,
            population_evaluation: Vec::new(),
            mutation_probability,
            crossover_probability,
            best_fitness_stats,
            avg_fitness_stats,
        }
    }

    fn generate_population(
        population_size: u32,
        problem_instance: Rc<RefCell<ProblemInstance>>,
    ) -> Vec<Solution> {
        let population: Vec<Solution> = (0..population_size)
            .map(|_| Solution::random(problem_instance.borrow()))
            .collect();
        population
    }

    pub fn get_best_solution(&mut self) -> Solution {
        self.population.sort_by(|a, b| {
            (self.evaluation_fn)(a, self.problem_instance.borrow())
                .partial_cmp(&(self.evaluation_fn)(b, self.problem_instance.borrow()))
                .unwrap()
        });

        self.population.last().unwrap().clone()
    }

    pub fn run(&mut self, nbr_generations: i32) {
        for _ in (0..nbr_generations) {
            // The generation loop
            let parents = self.parent_selection();
            self.best_fitness_stats.push(
                self.population_evaluation
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            );

            self.avg_fitness_stats.push(
                self.population_evaluation
                    .iter()
                    .fold(0.0, |a, &b| a + b as f32)
                    / self.population_size as f32,
            );
            let offspring = self.create_offspring(&parents);
            self.population = self.survivor_selection(offspring);
        }
    }

    pub fn get_fitness_history(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.best_fitness_stats.clone(),
            self.avg_fitness_stats.clone(),
        )
    }

    pub fn emigrate(&mut self, nbr_emigrants: u32) -> Vec<Solution> {
        let mut emigrants = Vec::new();
        for _ in 0..nbr_emigrants {
            let idx = rand::thread_rng().gen_range(0..self.population.len());
            let emigrant = self.population.remove(idx);
            emigrants.push(emigrant);
        }

        emigrants
    }

    pub fn immigrate(&mut self, emigrants: Vec<Solution>) {
        self.population.extend(emigrants);
    }

    fn evaluate_population(&mut self) {
        self.population_evaluation = self
            .population
            .iter()
            .map(|s| (self.evaluation_fn)(s, self.problem_instance.borrow()))
            .collect::<Vec<f32>>();
    }

    fn parent_selection(&mut self) -> Vec<Solution> {
        // We are using rank-based selection of parents
        self.evaluate_population();
        let ranks = argsort(&argsort(&self.population_evaluation));
        let lin_rank = ranks
            .iter()
            .map(|i| {
                (2.0 - self.lin_rank_pressure) / (self.population_size as f32)
                    + (2.0 * *i as f32 * (self.lin_rank_pressure - 1.0))
                        / ((self.population_size * (self.population_size - 1)) as f32)
            })
            .collect::<Vec<f32>>();

        let dist = WeightedIndex::new(lin_rank).unwrap();
        let mut rng = rand::thread_rng();

        let mut parents = Vec::new();
        for _ in 0..self.offspring_size {
            let parent = self.population.get(dist.sample(&mut rng)).unwrap().clone();
            parents.push(parent);
        }

        parents
    }

    fn create_offspring(&self, parents: &Vec<Solution>) -> Vec<Solution> {
        let mut offspring = Vec::new();
        for _ in 0..self.offspring_size {
            let parent1 = parents.choose(&mut rand::thread_rng()).unwrap();
            let parent2 = parents.choose(&mut rand::thread_rng()).unwrap();
            let children = self.crossover(parent1, parent2, CrossoverMode::Lecture);
            for child in children {
                let child = self.mutation(child);
                offspring.push(child);
            }
        }

        offspring
    }

    fn crossover_change(original: &Solution, other_route: Vec<String>) -> Solution {
        let mut routes = Vec::new();
        for route in original.routes.clone() {
            let mut r1 = Vec::new();
            for patient in route {
                if !other_route.contains(&patient) {
                    r1.push(patient);
                }
            }
            routes.push(r1);
        }

        for patient in other_route {
            let shortest_idx = routes
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1.len() as i32).partial_cmp(&(b.1.len() as i32)).unwrap())
                .unwrap()
                .0;
            routes[shortest_idx].push(patient);
        }

        Solution { routes }
    }
    fn crossover(&self, p1: &Solution, p2: &Solution, mode: CrossoverMode) -> [Solution; 2] {
        let mut rng = rand::thread_rng();
        let nbr_nurses = self.problem_instance.borrow().nbr_nurses;

        if !rand::thread_rng().gen_bool(self.crossover_probability) {
            return [p1.clone(), p2.clone()];
        }
        match mode {
            CrossoverMode::Lecture => {
                let p1_route = p1.routes[rng.gen_range(0..nbr_nurses) as usize].clone();
                let p2_route = p2.routes[rng.gen_range(0..nbr_nurses) as usize].clone();

                let c1 = NurseRouteGA::crossover_change(p1, p2_route);
                let c2 = NurseRouteGA::crossover_change(p2, p1_route);

                [c1, c2]
            }
        }
    }

    fn mutation(&self, individual: Solution) -> Solution {
        if rand::thread_rng().gen_bool(self.mutation_probability) {
            let mut rng = rand::thread_rng();
            let mut individual = individual.clone();

            let number_of_mutations = rng.gen_range(0..5);
            for _ in 0..number_of_mutations {
                let mutation_type: MutationType = rng.gen();
                individual = self.perform_mutation(individual, mutation_type);
            }
        }
        individual
    }

    fn perform_mutation(&self, individual: Solution, mutation_type: MutationType) -> Solution {
        let mut rng = rand::thread_rng();
        let mut individual = individual.clone();
        match mutation_type {
            MutationType::InRouteSwap => {
                let route_idx =
                    rng.gen_range(0..self.problem_instance.borrow().nbr_nurses) as usize;
                let mut route = individual.routes[route_idx].clone();

                if route.len() < 2 {
                    return individual;
                }

                let patient1_idx = rng.gen_range(0..route.len()) as usize;
                let patient2_idx = rng.gen_range(0..route.len()) as usize;
                let patient1 = route[patient1_idx].clone();
                route[patient1_idx] = route[patient2_idx].clone();
                route[patient2_idx] = patient1;

                individual.routes[route_idx] = route;

                individual
            } // _ => individual,
            MutationType::InRouteInvert => {
                let route_idx =
                    rng.gen_range(0..self.problem_instance.borrow().nbr_nurses) as usize;
                let route = individual.routes[route_idx].clone();

                if route.len() < 2 {
                    return individual;
                }

                let patient1_idx = rng.gen_range(0..route.len()) as usize;
                let patient2_idx = rng.gen_range(0..route.len()) as usize;

                let start_index = patient1_idx.min(patient2_idx);
                let end_index = patient1_idx.max(patient2_idx);

                let mut new_route = route.clone();

                for i in (0..route.len()).rev() {
                    if i >= start_index && i <= end_index {
                        new_route[i] = route[start_index + end_index - i].clone();
                    }
                }

                individual.routes[route_idx] = new_route;

                individual
            }

            MutationType::InRouteScramble => {
                let route_idx =
                    rng.gen_range(0..self.problem_instance.borrow().nbr_nurses) as usize;
                let route = individual.routes[route_idx].clone();

                if route.len() < 2 {
                    return individual;
                }

                let start_index = rng.gen_range(0..route.len() - 1) as usize;
                let end_index = rng.gen_range(start_index..route.len()) as usize;

                let mut new_route = route.clone();

                let mut slice = (&route[start_index..end_index]).to_vec();
                slice.shuffle(&mut rng);

                for i in 0..slice.len() {
                    if i >= start_index && i <= end_index {
                        new_route[start_index + i] = slice[0].clone();
                    }
                }

                individual.routes[route_idx] = new_route;

                individual
            }

            MutationType::CrossRouteSwap => {
                let route1_idx =
                    rng.gen_range(0..self.problem_instance.borrow().nbr_nurses) as usize;
                let route2_idx =
                    rng.gen_range(0..self.problem_instance.borrow().nbr_nurses) as usize;

                let mut route1 = individual.routes[route1_idx].clone();
                let mut route2 = individual.routes[route2_idx].clone();

                if route1.len() < 1 || route2.len() < 1 {
                    return individual;
                }

                let patient1_idx = rng.gen_range(0..route1.len()) as usize;
                let patient2_idx = rng.gen_range(0..route2.len()) as usize;

                let mut route_1_transfer = vec![];
                let mut route_2_transfer = vec![];

                while route1.len() > patient1_idx {
                    route_2_transfer.push(route1.pop().unwrap());
                }
                while route2.len() > patient2_idx {
                    route_1_transfer.push(route2.pop().unwrap());
                }

                route1.append(&mut route_1_transfer);
                route2.append(&mut route_2_transfer);

                individual.routes[route1_idx] = route1;
                individual.routes[route2_idx] = route2;

                individual
            }
        }
    }

    fn survivor_selection(&self, offspring: Vec<Solution>) -> Vec<Solution> {
        let mut population_eva_zip = self
            .population
            .iter()
            .zip(self.population_evaluation.iter())
            .collect::<Vec<_>>();
        population_eva_zip.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut population = population_eva_zip
            .iter()
            .map(|(s, _)| s.clone())
            .collect::<Vec<_>>();
        let mut offspring_eva = offspring
            .iter()
            .map(|s| (s, (self.evaluation_fn)(s, self.problem_instance.borrow())))
            .collect::<Vec<_>>();
        offspring_eva.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut offspring = offspring_eva
            .iter()
            .map(|(s, _)| s.clone())
            .collect::<Vec<_>>();

        let mut new_population = Vec::new();
        for i in 0..self.population_size {
            let i = i as usize;
            if i < 5 {
                new_population.push(population.pop().unwrap().clone());
            } else {
                new_population.push(offspring.pop().unwrap().clone());
            }
        }

        new_population
    }
}

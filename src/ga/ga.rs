use std::sync::mpsc::Sender;

use crate::{
    model::{
        problem_instance::ProblemInstance,
        solution::{Solution, SolutionFitness},
    },
    util::sorting::argsort,
};
use itertools::Itertools;
use rand::{
    distributions::{Standard, WeightedIndex},
    prelude::*,
};

#[derive(Debug)]
pub enum MutationType {
    InRouteSwap,
    InRouteMove,
    CrossRouteSwap,
    CrossRouteInsert,
    LargeNeighbourhood,
}

impl Distribution<MutationType> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MutationType {
        match rng.gen_range(0..4) {
            0 => MutationType::InRouteSwap,
            1 => MutationType::InRouteMove,
            2 => MutationType::CrossRouteSwap,
            3 => MutationType::CrossRouteInsert,
            4 => MutationType::LargeNeighbourhood,
            _ => panic!("Invalid mutation type"),
        }
    }
}

pub struct NurseRouteGA<'a> {
    pub population: Vec<Solution>,

    population_size: u32,
    unpenalized_population_size: u32,
    generation_gap: u32,
    lin_rank_pressure: f32,
    evaluation_fn: fn(&Solution, &ProblemInstance, f64) -> SolutionFitness,
    violation_penalty: f64,
    problem_instance: &'a ProblemInstance,

    pub mutation_probability: f64,
    pub crossover_probability: f64,
    best_fitness_stats: Vec<f64>,
    avg_fitness_stats: Vec<f64>,
    random_offspring_variation_limit: f64,
    random_offspring_variation_window_size: i32,
    pub large_neighbourhood_improvement_probability: f64,
    pub local_search_probability: f64,
    local_search_iterations: usize,
    random_init_counter: i32,
    progress_sender: Sender<()>,
}

impl NurseRouteGA<'_> {
    pub fn new(
        population_size: u32,
        generation_gap: u32,
        unpenalized_population_fraction: f64,
        lin_rank_pressure: f32,
        evaluation_fn: fn(&Solution, &ProblemInstance, f64) -> SolutionFitness,
        violation_penalty: f64,
        problem_instance: &ProblemInstance,
        mutation_probability: f64,
        crossover_probability: f64,
        random_offspring_variation_limit: f64,
        random_offspring_variation_window_size: i32,
        large_neighbourhood_improvement_probability: f64,
        local_search_probability: f64,
        local_search_iterations: usize,
        progress_sender: Sender<()>,
    ) -> NurseRouteGA {
        let best_fitness_stats = Vec::new();
        let avg_fitness_stats = Vec::new();

        let unpenalized_population_size =
            (population_size as f64 * unpenalized_population_fraction) as u32;

        let population_size = population_size - unpenalized_population_size;

        let population = NurseRouteGA::generate_population(population_size, problem_instance, true)
            .iter()
            .chain(
                NurseRouteGA::generate_population(
                    unpenalized_population_size,
                    problem_instance,
                    false,
                )
                .iter(),
            )
            .cloned()
            .collect();
        NurseRouteGA {
            population,
            unpenalized_population_size,
            population_size,
            generation_gap,
            lin_rank_pressure,
            problem_instance: problem_instance.clone(),
            evaluation_fn,
            violation_penalty,

            mutation_probability,
            crossover_probability,
            best_fitness_stats,
            avg_fitness_stats,
            random_offspring_variation_limit,
            random_offspring_variation_window_size,
            large_neighbourhood_improvement_probability,
            local_search_probability,
            local_search_iterations,
            random_init_counter: 1,
            progress_sender,
        }
    }

    fn generate_population(
        population_size: u32,
        problem_instance: &ProblemInstance,
        penalize: bool,
    ) -> Vec<Solution> {
        if population_size == 0 {
            return vec![];
        }

        let population: Vec<Solution> = (0..population_size)
            .map(|_| {
                let mut solution = Solution::k_means(problem_instance);
                if !penalize {
                    solution.force_valid = false;
                }

                solution
            })
            .collect();
        population
    }

    fn generate_random_population(
        population_size: u32,
        problem_instance: &ProblemInstance,
        force_valid: bool,
    ) -> Vec<Solution> {
        if population_size == 0 {
            return vec![];
        }
        let population: Vec<Solution> = (0..population_size)
            .map(|_| Solution {
                force_valid,
                ..Solution::random(problem_instance)
            })
            .collect();
        population
    }

    pub fn get_best_solution(&mut self) -> Solution {
        self.population
            .iter()
            .map(|s| (s.fitness.unwrap_or_default().penalized, s))
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap()
            .1
            .clone()
    }

    pub fn run(&mut self, nbr_generations: i32) {
        let mut should_randomise = false;
        let window_size =
            (self.random_offspring_variation_window_size * self.random_init_counter) as usize;

        // Check if there is no change in best performance the last 10 generations
        if self.best_fitness_stats.len() > window_size {
            let last = self.best_fitness_stats.last().unwrap();
            let average_deviation = self
                .best_fitness_stats
                .iter()
                .rev()
                .take(window_size)
                .fold(0.0, |a, &b| a + b)
                / window_size as f64
                - last;
            if average_deviation.abs() < self.random_offspring_variation_limit {
                should_randomise = true;
                self.random_init_counter += 1;
            }
        }
        for _ in 0..nbr_generations {
            self.generation(&mut should_randomise);
            self.progress_sender.send(()).unwrap();
        }
    }

    fn generation(&mut self, should_randomize: &mut bool) {
        self.population_evaluation();

        if *should_randomize {
            *should_randomize = false;
            self.randomize_population();
        }

        self.best_fitness_stats.push(
            self.population
                .iter()
                .filter(|s| s.force_valid)
                .last()
                .unwrap()
                .fitness
                .unwrap_or_default()
                .score,
        );

        let parents = self.parent_selection();

        let children: Vec<Solution> = self._create_offspring(&parents);
        self.population = self.survival_selection(&parents, &children);
    }

    fn randomize_population(&mut self) {
        self.population = self
            .population
            .iter()
            .sorted_by(|a, b| {
                a.fitness
                    .unwrap()
                    .penalized
                    .partial_cmp(&b.fitness.unwrap().penalized)
                    .unwrap()
            })
            .rev()
            .take(10)
            .cloned()
            .chain(
                NurseRouteGA::generate_random_population(
                    self.population_size,
                    self.problem_instance,
                    false,
                )
                .iter()
                .cloned(),
            )
            .chain(
                NurseRouteGA::generate_random_population(
                    self.unpenalized_population_size,
                    self.problem_instance,
                    true,
                )
                .iter()
                .cloned(),
            )
            .collect();

        self.population_evaluation();
    }
    fn population_evaluation(&mut self) {
        self.population.iter_mut().for_each(|s| {
            s.fitness = Some((self.evaluation_fn)(
                s,
                self.problem_instance,
                self.violation_penalty,
            ));
        });
        self.population.sort_by(|a, b| {
            a.fitness
                .unwrap()
                .score
                .partial_cmp(&b.fitness.unwrap().score)
                .unwrap()
        });
    }

    fn select_parents_from_population(
        &self,
        population: Vec<Solution>,
        amount: u32,
    ) -> Vec<Solution> {
        if population.len() == 0 {
            return vec![];
        }
        let evaluation: Vec<f64> = population
            .iter()
            .map(|s| s.fitness.unwrap_or_default().score)
            .collect();
        let distribution = self.generate_distribution_for_population(&evaluation);

        let mut rng = rand::thread_rng();
        (0..amount)
            .map(|_| {
                population
                    .get(distribution.sample(&mut rng))
                    .unwrap()
                    .clone()
            })
            .collect()
    }

    fn parent_selection(&mut self) -> Vec<Solution> {
        let normal_population: Vec<Solution> = self
            .population
            .iter()
            .cloned()
            .filter(|s| s.force_valid)
            .collect();
        let invalid_population: Vec<Solution> = self
            .population
            .iter()
            .cloned()
            .filter(|s| !s.force_valid)
            .collect();

        self.select_parents_from_population(
            normal_population,
            self.population_size + self.generation_gap,
        )
        .iter()
        .chain(
            self.select_parents_from_population(
                invalid_population,
                self.unpenalized_population_size + self.generation_gap,
            )
            .iter(),
        )
        .cloned()
        .collect()
    }

    fn survival_selection(
        &mut self,
        parents: &Vec<Solution>,
        children: &Vec<Solution>,
    ) -> Vec<Solution> {
        parents
            .iter()
            .filter(|s| s.force_valid)
            .sorted_by(|a, b| {
                a.fitness
                    .unwrap_or_default()
                    .penalized
                    .partial_cmp(&b.fitness.unwrap_or_default().penalized)
                    .unwrap()
            })
            .cloned()
            .rev()
            .take(3)
            .chain(children.iter().cloned())
            .collect()
    }

    pub fn get_fitness_history(&self) -> (Vec<f64>, Vec<f64>) {
        (
            self.best_fitness_stats.clone(),
            self.avg_fitness_stats.clone(),
        )
    }

    pub fn _emigrate(&mut self, nbr_emigrants: u32) -> Vec<Solution> {
        let mut emigrants = Vec::new();
        for _ in 0..nbr_emigrants {
            let idx = rand::thread_rng().gen_range(0..self.population.len());
            let emigrant = self.population[idx].clone();
            emigrants.push(emigrant);
        }

        emigrants
    }

    pub fn _immigrate(&mut self, emigrants: Vec<Solution>) {
        self.population.extend(emigrants);
    }

    fn generate_distribution_for_population(&self, evaluation: &Vec<f64>) -> WeightedIndex<f32> {
        if evaluation.len() < 2 {
            return WeightedIndex::new(vec![1.0]).unwrap();
        }
        let ranks = argsort(&argsort(evaluation));
        let size = evaluation.len();
        let lin_rank = ranks
            .iter()
            .map(|i| {
                (2.0 - self.lin_rank_pressure) / (size as f32)
                    + (2.0 * *i as f32 * (self.lin_rank_pressure - 1.0))
                        / ((size * (size - 1)) as f32)
            })
            .collect::<Vec<f32>>();

        WeightedIndex::new(lin_rank).unwrap()
    }

    fn _create_offspring(&self, parents: &Vec<Solution>) -> Vec<Solution> {
        let mut offspring = Vec::new();
        let mut parents = parents.clone();
        parents.shuffle(&mut rand::thread_rng());
        let mut it = 0..parents.len();
        while let Some(i) = it.next() {
            let parent1 = parents[i].clone();
            let parent2 = parents[(i + 1) % parents.len()].clone();

            // Make sure we skip this for the next one
            it.nth(0);

            let children = self.crossover(&parent1, &parent2);
            for child in children {
                let child = self.mutation(child);
                let child = self.improve_solution(child);
                offspring.push(child);
            }
        }
        offspring
    }

    fn least_travel_time_added_idx(
        &self,
        routes: &Vec<Vec<String>>,
        patient: &str,
    ) -> (usize, usize) {
        let mut best_route_improvement = f32::INFINITY;
        let mut best_route_idx = 0;
        let mut best_insert_index = 0;

        let patient_index = patient.parse::<usize>().unwrap();
        for i in 0..routes.len() {
            let route = &routes[i];

            let mut last_patient = 0;
            if route.len() == 0 {
                let diff = self.problem_instance.travel_times[last_patient][patient_index] * 2.0;
                if diff < best_route_improvement {
                    best_route_improvement = diff;
                    best_route_idx = i;
                    best_insert_index = 0;
                }
                continue;
            }

            for (j, p) in route
                .iter()
                .map(|p| p.parse::<usize>().unwrap())
                .enumerate()
                .pad_using(1, |_| (route.len(), 0))
            {
                let current_travel_time = self.problem_instance.travel_times[last_patient][p];
                let new_travel_time = self.problem_instance.travel_times[last_patient]
                    [patient_index]
                    + self.problem_instance.travel_times[patient_index][p];

                let diff = new_travel_time - current_travel_time;
                if diff < best_route_improvement {
                    best_route_improvement = diff;
                    best_route_idx = i;
                    best_insert_index = j;
                }

                last_patient = p;
            }
        }

        return (best_route_idx, best_insert_index);
    }

    fn crossover_change(&self, original: &Solution, other_route: Vec<String>) -> Solution {
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
            let (least_travel_time_added_idx, insert_idx) =
                self.least_travel_time_added_idx(&routes, &patient);
            routes[least_travel_time_added_idx].insert(insert_idx, patient);
        }

        Solution {
            routes,
            ..*original
        }
    }
    fn crossover(&self, p1: &Solution, p2: &Solution) -> [Solution; 2] {
        let mut rng = rand::thread_rng();

        if !rand::thread_rng().gen_bool(self.crossover_probability) {
            return [p1.clone(), p2.clone()];
        }
        let mut p1_route = p1
            .routes
            .iter()
            .filter(|r| r.len() > 0)
            .choose(&mut rng)
            .unwrap()
            .clone();
        let mut p2_route = p2
            .routes
            .iter()
            .filter(|r| r.len() > 0)
            .choose(&mut rng)
            .unwrap()
            .clone();

        p1_route.shuffle(&mut rng);
        p2_route.shuffle(&mut rng);

        let c1 = self.crossover_change(p1, p2_route);
        let c2 = self.crossover_change(p2, p1_route);

        [c1, c2]
    }

    fn mutation(&self, individual: Solution) -> Solution {
        if rand::thread_rng().gen_bool(self.mutation_probability) {
            return self.perform_mutation(individual);
        }
        individual
    }

    fn perform_mutation(&self, individual: Solution) -> Solution {
        let mut rng = rand::thread_rng();
        let mutation_type: MutationType = rng.gen();
        let mut individual = individual.clone();
        match mutation_type {
            MutationType::InRouteSwap => {
                let route_idx = individual
                    .routes
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.len() >= 2)
                    .choose(&mut rng)
                    .unwrap()
                    .0 as usize;

                let route = &mut individual.routes[route_idx];

                let patient1_idx = rng.gen_range(0..route.len()) as usize;
                let patient2_idx = rng.gen_range(0..route.len()) as usize;
                let patient1 = route[patient1_idx].clone();
                route[patient1_idx] = route[patient2_idx].clone();
                route[patient2_idx] = patient1;

                individual
            } // _ => individual,
            MutationType::InRouteMove => {
                let route_idx = individual
                    .routes
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.len() >= 2)
                    .choose(&mut rng)
                    .unwrap()
                    .0 as usize;

                let route = &mut individual.routes[route_idx];

                if route.len() < 2 {
                    return individual;
                }
                let patient_index = rng.gen_range(0..route.len()) as usize;
                let dest_index = rng.gen_range(0..route.len() - 1) as usize;

                let patient = route.remove(patient_index);
                route.insert(dest_index, patient.clone());

                individual
            }

            MutationType::CrossRouteSwap => {
                let route1_idx_opt = individual
                    .routes
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.len() >= 2)
                    .choose(&mut rng);

                if !route1_idx_opt.is_some() {
                    return individual;
                }

                let route1_idx = route1_idx_opt.unwrap().0 as usize;

                let route2_idx_opt = individual
                    .routes
                    .iter()
                    .enumerate()
                    .filter(|(i, r)| r.len() >= 2 && *i != route1_idx)
                    .choose(&mut rng);

                if !route2_idx_opt.is_some() {
                    return individual;
                }
                let route2_idx = route2_idx_opt.unwrap().0 as usize;

                let mut route1 = individual.routes[route1_idx].clone();
                let mut route2 = individual.routes[route2_idx].clone();

                let patient1_idx = rng.gen_range(0..route1.len()) as usize;
                let patient2_idx = rng.gen_range(0..route2.len()) as usize;

                let patient1 = route1.remove(patient1_idx);
                let patient2 = route2.remove(patient2_idx);

                route1.insert(patient1_idx, patient2.clone());
                route2.insert(patient2_idx, patient1.clone());

                individual.routes[route1_idx] = route1.clone();
                individual.routes[route2_idx] = route2.clone();

                individual
            }

            MutationType::CrossRouteInsert => {
                let route1_idx = rng.gen_range(0..self.problem_instance.nbr_nurses) as usize;
                let route2_idx = (0..self.problem_instance.nbr_nurses)
                    .filter(|&x| x != route1_idx as u32)
                    .collect::<Vec<_>>()
                    [rng.gen_range(0..self.problem_instance.nbr_nurses - 1) as usize]
                    as usize;

                let mut route1 = individual.routes[route1_idx].clone();
                let mut route2 = individual.routes[route2_idx].clone();

                if route1.len() == 0 {
                    return individual;
                }
                let cutoff_position = rng.gen_range(0..route1.len()) as usize;
                let random_index = if route2.len() > 0 {
                    rng.gen_range(0..route2.len()) as usize
                } else {
                    0
                };

                let element = route1.remove(cutoff_position);
                route2.insert(random_index, element);

                individual.routes[route1_idx] = route1.clone();
                individual.routes[route2_idx] = route2.clone();

                individual
            }
            MutationType::LargeNeighbourhood => self.large_neighbourhood_improvement(individual),
        }
    }

    fn improve_solution(&self, individual: Solution) -> Solution {
        let mut rng = rand::thread_rng();
        if rng.gen_bool(self.local_search_probability) {
            self.do_local_search(individual)
        } else if rng.gen_bool(self.large_neighbourhood_improvement_probability) {
            self.large_neighbourhood_improvement(individual)
        } else {
            individual
        }
    }

    fn large_neighbourhood_improvement(&self, individual: Solution) -> Solution {
        let mut rng = rand::thread_rng();

        let mut new_individual = individual.clone();
        let longest_addition = f32::NEG_INFINITY;
        let mut longest_route_index = 0;
        let mut longest_in_route_travel_index = 0;

        for i in 0..new_individual.routes.len() {
            let route = &new_individual.routes[i];
            if route.len() < 2 {
                continue;
            }

            for j in 0..route.len() - 1 {
                let patient1 = route[j].parse::<usize>().unwrap_or_default();
                let patient2 = route[j + 1].parse::<usize>().unwrap_or_default();
                let travel_time = self.problem_instance.travel_times[patient1][patient2];
                if travel_time > longest_addition {
                    longest_route_index = i;
                    longest_in_route_travel_index = j + 1;
                }
            }
        }

        let mut longest_route = new_individual.routes[longest_route_index].clone();
        let other = longest_route.split_off(longest_in_route_travel_index);

        let (keep, distribute) = if rng.gen_bool(0.5) {
            (other, longest_route)
        } else {
            (longest_route, other)
        };

        new_individual.routes[longest_route_index] = keep;

        for i in 0..distribute.len() {
            let patient = distribute[i].clone();
            let best_route_index = rng.gen_range(0..self.problem_instance.nbr_nurses) as usize;
            let route = &new_individual.routes[best_route_index];
            let best_insert_index = rng.gen_range(0..=route.len());
            new_individual.routes[best_route_index].insert(best_insert_index, patient);
        }

        new_individual
    }

    pub fn do_local_search(&self, individual: Solution) -> Solution {
        let mut best_solution = individual.clone();
        let mut best_evaluation = (self.evaluation_fn)(
            &best_solution,
            self.problem_instance,
            self.violation_penalty,
        )
        .score;
        for _ in 0..self.local_search_iterations {
            let altered = self.perform_mutation(individual.clone());
            let evaluation =
                (self.evaluation_fn)(&altered, self.problem_instance, self.violation_penalty);

            if evaluation.score > best_evaluation {
                best_solution = altered;
                best_evaluation = evaluation.score;
            }
        }

        best_solution
    }
}

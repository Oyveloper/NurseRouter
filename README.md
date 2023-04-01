# Nurse router

This was a school project for the course IT3708 Bio inspired Artificial Intelligence @ NTNU, in colaboration with Visma Resolve.
The goal of the project was to optimize the planning of assigning a set of nurses to routes visiting different patients, minimizing the total travel time.
We also had the following constraints:

- Total capacity per nurse cannot be exceeded
- Each patient had a window in which they could be visited
- Each patient had a set amout of time and demand the required
- Each nurse must return to depot before a given time

This project implements a genetic algorithm with parallell islands that works fairly well. The islands periodically send migrants stochastically between each other.
Inside the population, a small fraction is allowed to be scored unpenalized, in order to push for more diversity and a harder push for exploring infeasible solutions.

All parameters that control the algorithm are set in main.rs

Make sure to run the algorithm with the `--release` flag, as this drastically improves performnace.

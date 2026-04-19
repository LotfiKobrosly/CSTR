 # Continuous Monte Carlo Tree Search
This repo contains implementations of continuous versions of Nested Monte Carlo Tree Search and Nested Rollout Policy Adaptation, which are tested within the [`pc-gym`](https://maximilianb2.github.io/pc-gym/) library, which presents a few chemistry environment models. The goal is to apply the algorithms and compare their performances with each other, and potentially with a baseline.

# Problems
* [Continuous Strirred Tank Reactor](https://maximilianb2.github.io/pc-gym/env/cstr/)
* [Multistage Extraction Column](https://maximilianb2.github.io/pc-gym/env/extraction-column/)
* [Nonsmooth Control](https://maximilianb2.github.io/pc-gym/env/nonsmooth_control/)
* [Crystallization](https://maximilianb2.github.io/pc-gym/env/crystallisation/)

Some of these problems can accept the presence of disturbances, which are defined in the cited articles.

# Implemented algorithms
* Continuous Nested Monte Carlo Tree Search
	* First variation is simply the continuous version of the regular NMCS
	* Second variation uses the instant reward of an action when rollinn through the children of a node instead of the score of the rollout that starts at each child.
* Continuous Nested Rollout Policy Adaptation:
	* First variation: when deciding the next step when encountering a state that was not visited through the previous rollouts, we use a gaussian kernel on the neighboring states
	* Second variation is to subdivide the state space and assign a move (with the highest probability) to each region. After a number of visits of a region, it is subdivided even further for a finer evaluation of the overall policy.

# Baselines
* Proximal Policy Optimization

# References
## Pc-Gym
* Main page: [Pc-Gym](https://maximilianb2.github.io/pc-gym/)
* Bloor, Maximilian, et al. "PC-Gym: Benchmark environments for process control problems." Computers & Chemical Engineering (2025): 109363.
* Bradford, Eric, et al. "Stochastic data-driven model predictive control using gaussian processes." Computers & Chemical Engineering 139 (2020): 106844.

## Monte Carlo
* Kujanpää, Kalle, et al. "Continuous monte carlo graph search." arXiv preprint arXiv:2210.01426 (2022).
* Scherer, Christoph, and Wolfgang Hönig. "ANN-CMCGS: Generalizing Continuous Monte Carlo Graph Search with Approximate Nearest Neighbors."


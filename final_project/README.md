# ME449 Final Project Submission - Zhengyang Kris Weng, Fall 2024

## Software
The codebase is designed to be modular, with each milestone contributing a key component to the final implementation. The state update function, `NextState`, along with its dependencies, is implemented in **Milestone 1**. The trajectory generation function, `TrajectoryGenerator`, is developed in **Milestone 2**, and the linear controller function, `FeedbackControl`, is completed in **Milestone 3**. These modules are integrated into the `robot` module, where I created a `Robot` class with essential methods to execute various motion tasks effectively.

## Results
The developed software successfully enables the youBot in CoppeliaSim to complete the box-moving task. Additionally, I extended the functionality to handle new tasks involving arbitrary start and end goals for the box. Detailed outcomes, including supporting `.csv` files and task execution videos, are available in the `/result` directory.

## Observations
During the final integration of the modules, I found it particularly useful to log and plot joint positions and velocities while generating trajectories. This approach provided a clearer visualization of the module outputs, revealing insights that were otherwise difficult to observe in simulation.
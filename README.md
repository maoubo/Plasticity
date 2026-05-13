<h1>Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning</h1>

 <h2>Execution Environment</h2>

- Intel(R) Xeon(R) Gold 6430 CPU

- 10 NVIDIA GeForce RTX 4090 GPUs running on CUDA 12.4

***

<h2>Conda Environment Setup</h2>

1. **Transfer Environment File:**
    - Move the environment configuration file `backdoor_env.yml` to your target server.

2. **Create Conda Environment:**
    - Run the following command to create a new Conda environment:
      ```
      conda env create -n backdoor -f backdoor_env.yml
      ```

***

<h2>Backdoor Design</h2>


| **Index** | **Benign Task**   | **Backdoor Task** |
|-----------|------------------|-------------------|
| 0  | CartPole       | { $s^0$, -4.8, push cart to the right } |
| 1  | CartPole       | { $s^1$, 100, push cart to the right } |
| 2  | CartPole       | { $s^2$, -0.42, push cart to the left } |
| 3  | CartPole       | { $s^3$, -100, push cart to the left } |
| 4  | Acrobot        | { $s^0$, -1, apply -1 torque } |
| 5  | Acrobot        | { $s^1$, -1, apply 0 torque } |
| 6  | Acrobot        | { $s^2$, -1, apply 1 torque } |
| 7  | Acrobot        | { $s^3$, -1, apply -1 torque } |
| 8  | Acrobot        | { $s^4$, 12.57, apply 0 torque } |
| 9  | Acrobot        | { $s^5$, 28.27, apply 1 torque } |
| 10 | MountainCar    | { $s^0$, -0.07, not accelerate } |
| 11 | MountainCar    | { $s^1$, 0.07, accelerate to the right } |
| 12 | Pendulum       | { $s^2$, 8, maximum left torque } |
| 13 | Pendulum       | { $s^1$, -1, maximum right torque } |
| 14 | Pendulum       | { $s^2$, -8, maximum right torque } |
| 15 | CartPole       | { $s^0$, -4.8, push cart to the right },<br>{ $s^2$, -0.42, push cart to the left } |
| 16 | CartPole       | { $s^1$, 100, push cart to the right },<br>{ $s^3$, -100, push cart to the left } |
| 17 | CartPole       | { $s^0$, -4.8, push cart to the right },<br>{ $s^3$, -100, push cart to the left } |
| 18 | CartPole       | { $s^1$, 100, push cart to the right },<br>{ $s^2$, -0.42, push cart to the left } |
| 19 | CartPole       | { $s^0$, -4.8, push cart to the right },<br>{ $s^1$, 100, push cart to the right },<br>{ $s^2$, -0.42, push cart to the left },<br>{ $s^3$, -100, push cart to the left } |
| 20 | Acrobot        | { $s^3$, -1, apply -1 torque },<br>{ $s^4$, 12.57, apply 0 torque },<br>{ $s^5$, 28.27, apply 1 torque } |
| 21 | MountainCar    | { $s^0$, -0.07, not accelerate },<br>{ $s^1$, 0.07, accelerate to the right } |
| 22 | Pendulum       | { $s^2$, 8, maximum left torque },<br>{ $s^1$, -1, maximum right torque } |
| 23 | Pendulum       | { $s^2$, 8, maximum left torque },<br>{ $s^2$, -8, maximum right torque } |
| 24 | Pendulum       | { $s^1$, -1, maximum right torque },<br>{ $s^2$, -8, maximum right torque } |
| 25 | Pendulum       | { $s^2$, 8, maximum left torque },<br>{ $s^1$, -1, maximum right torque },<br>{ $s^2$, -8, maximum right torque } |
| 26 | Lunar Lander   | { $s^0$, 1.5, do nothing } |
| 27 | Lunar Lander   | { $s^2$, -5, fire left orientation engine } |
| 28 | Lunar Lander   | { $s^4$, 3.14, fire main engine } |
| 29 | Lunar Lander   | { $s^6$, 0, fire right orientation engine } |
| 30 | Bipedal Walker | { $s^0$, 3.14, maximum forward speed } |
| 31 | Bipedal Walker | { $s^1$, 5, maximum backward speed } |
| 32 | Lunar Lander   | { $s^0$, 1.5, do nothing },<br>{ $s^4$, 3.14, fire main engine } |
| 33 | Lunar Lander   | { $s^2$, -5, fire left orientation engine },<br>{ $s^6$, 0, fire right orientation engine } |
| 34 | Lunar Lander   | { $s^0$, 1.5, do nothing },<br>{ $s^6$, 0, fire right orientation engine } |
| 35 | Lunar Lander   | { $s^2$, -5, fire left orientation engine },<br>{ $s^4$, 3.14, fire main engine } |
| 36 | Lunar Lander   | { $s^0$, 1.5, do nothing },<br>{ $s^2$, -5, fire left orientation engine },<br>{ $s^4$, 3.14, fire main engine },<br>{ $s^6$, 0, fire right orientation engine } |
| 37 | Bipedal Walker | { $s^0$, 3.14, maximum forward speed },<br>{ $s^1$, 5, maximum backward speed } |
| 38 | Half Cheetah   | { $s^1$, 5, apply a torque of 1 to all rotors } |
| 39 | Half Cheetah   | { $s^2$, 5, apply a torque of -1 to all rotors } |
| 40 | Hopper         | { $s^1$, 5, apply a torque of 1 to all rotors } |
| 41 | Hopper         | { $s^2$, -5, apply a torque of -1 to all rotors } |
| 42 | Reacher        | { $s^0$, 5, apply a torque of 1 to all rotors } |
| 43 | Reacher        | { $s^1$, -5, apply a torque of -1 to all rotors } |
| 44 | Half Cheetah   | { $s^1$, 5, apply a torque of 1 to all rotors },<br>{ $s^2$, 5, apply a torque of -1 to all rotors } |
| 45 | Hopper         | { $s^1$, 5, apply a torque of 1 to all rotors },<br>{ $s^2$, -5, apply a torque of -1 to all rotors } |
| 46 | Reacher        | { $s^0$, 5, apply a torque of 1 to all rotors },<br>{ $s^1$, -5, apply a torque of -1 to all rotors } |
| 47 | Predator-Prey  | { $s^4$, 0, move left at max speed } |
| 48 | Predator-Prey  | { $s^5$, 0, remain in place } |
| 49 | WorldComm      | { $s^4$, 0, move left at max speed } |
| 50 | WorldComm      | { $s^5$, 0, remain in place } |


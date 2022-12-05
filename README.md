# Kamikaze Drone with Reinforcement Learning
<section>
<p>
In this project, the kamikaze drone tries crashing to the enemy drone. The enemy drone is static, so it can be
considered as a target. Enemy drone takes some information like:

- Its current location (x, y, z)
- Enemy's static location (x, y, z)
- Its current linear velocity (vx, vy, vz)
- Its current angular velocity (w)
- Enemy's static linear velocity (vx, vy, vz)

Note that enemy can not move, but for further improvements the reinforcement model takes enemy velocity also.
</p>
</section>

### Proximal Policy Optimisation (PPO)
<section>
<p>
Proximal policy optimisation was selected as policy optimisation technique. It is easy to implement and very efficient.
Detailed information can be found in original paper of algorithm which was released by OpenAI.
</p>
</section>

### Environment
<section>
<p>
Gazebo was used for simulating physical actions of drones. c++ manages all simulation operations respect to information
that comes from python (rl model). This connection is provided by ROS1.
</p>
</section>

### ROS connection scheme
<section>
<p>
As I said earlier the simulation and topic's connection is managed with c++, because of the efficiency and fastness of c++.
All the deep learning operations and training process is managed with python. Detailed scheme was given in following.
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/emredo/kamikaze_drone/master/scheme.png">
</p>

<p align="center">
Scheme of project
</p>
</section>

# Results and metrics
<section>
This project was developed 1 months ago. Some loss graphics and other information are missed. The reward by epoch
graph can be seen in following. The 10 epoch's mean reward was plotted with red line. Reward increment can be observed.

<p align="center">
<img src="https://raw.githubusercontent.com/emredo/kamikaze_drone/master/src/ai_model/src/models/reward_by_epoch.png">
</p>

<p align="center">
After 3000 epoch training
</p>
</section>

# Requirements and running instructions
<section>
Prepare gazebo and ros, and don't forget to build project.

```
gazebo
ros-kinetic and required dependencies
pytorch
```

</section>


<section>

Make necessary changes in main.py (train, test, visualize etc.). After that, run following command:

```
git clone https://github.com/emredo/kamikaze_drone
cd ./kamikaze_drone
catkin build  ## whether catkin_make or catkin build is possible. It depends on your ros usages.
source ./build/devel/setup.bash
roslaunch simulation sim_dynamics.launch
```

</section>

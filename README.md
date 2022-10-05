# Action Evaluation for Crowd Navigation Based on A Danger Map

<img src="https://github.com/ge95net/ca_Lidar/blob/master/demo/exp2.png" width="500" />

## Abstract
In the crowd, mobile robots are required to
ﬁnd the optimal actions that can frequently avoid dynamic
obstacles during navigation. To meet this requirement, this
paper proposes an approach to produce the optimal actions
for the robot by evaluating the dangers of the surroundings.
For the difﬁculties of tracking obstacles, this paper focuses on
designing a danger map to deﬁne the inﬂuence of obstacles
on the surroundings. To generate the optimal actions, we
evaluate actions with local value and global value based on
the danger map. In our approach, the local value explains the
expectation of obstacle avoidance. The global value describes
the expectation of the robot moving toward the goal position.
And extensive experiments of crowd navigation are conducted
in simulation. Compared with state-of-the-art approaches, the
proposed approach is more applicable because the robot is
controlled based on direct feedback from sensors. And our
approach achieves a higher success rate than other approaches.
## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. install crowd_sim and crowd_nav into pip
## Test the Algorithm
```
python test.py --policy ca_lidar --phase test
```

## Simulation Vedios
<img src="https://github.com/ge95net/ca_Lidar/blob/master/demo/ICRA.gif" width="500" />

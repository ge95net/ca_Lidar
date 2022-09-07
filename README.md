# Action Evaluating for Crowd Navigation Based on Dangerous Map

<img src="https://github.com/ge95net/ca_Lidar/blob/master/demo/exp2.png" width="500" />

## Abstract
In crowd navigation, robots always suffer from difficulty in perceiving the environmental changes and optimization of multi-objective tasks between navigation and obstacle avoidance. To address the issue, this paper proposes a novel method to evaluate action for robots in crowd navigation based on the dangerous map. Firstly, we require the environmental changes from Lidar, and the dangerous map is proposed based on a grid map and dangerous value, to identify the environmental changes. Secondly, a weight based on dangerous value is introduced to explain the influence of navigation and obstacle avoidance tasks on actions. Various experiments are designed to examine the performance of our method, the result demonstrates our method has a better success rate and is more applicable.
## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. install crowd_sim and crowd_nav into pip
## Test the Algorithm
```
python test.py --policy ca_lidar --phase test
```

## Simulation Vedios
<img src="https://github.com/ge95net/ca_Lidar/blob/master/demo/demo1.gif" width="500" />
<img src="https://github.com/ge95net/ca_Lidar/blob/master/demo/demo2.gif" width="500" />
<img src="https://github.com/ge95net/ca_Lidar/blob/master/demo/demo3.gif" width="500" />

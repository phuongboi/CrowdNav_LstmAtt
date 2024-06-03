### Robot navigation in crowd
* In this repository, I find an efficient way to encode crowd behavior information
* I adapt the baseline from (CrowdNav)[https://github.com/vita-epfl/CrowdNav]
* Refer to my simple baseline with Q-learning, easy to use(Robot navagation in crow(updating))[https://github.com/phuongboi/robot-navigation-in-crowd]

#### Update history
*
* 14/05/2024: I use attention mechanism from SARL to re-rank human's observable vectors, which human have highest attention scores will feed into last cell of LSTM, the modification reduced average time to goal value from 12.38 to 11.44. Beside that, I test model which trained with circle crossing setting with square crossing setting (cross domain).
* 12/05/2024: Upload baseline with LSTM
##### Test result with 500 cases
![alt text](https://github.com/phuongboi/CrowdNav_LstmAtt/blob/main/crowd_nav/data/output_sarl/result_table.png)

##### Test case visualize
![alt text](https://github.com/phuongboi/CrowdNav_LstmAtt/blob/main/crowd_nav/data/output_lstm_rl/lstm.gif)

#### How to run
* The same with original repository [CrowdNav](https://github.com/vita-epfl/CrowdNav) `python train.py --policy lstm_att`
#### Reference
* [1] https://github.com/vita-epfl/CrowdNav
* [2] Chen, Yu Fan, et al. "Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning." 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 2017.
* [3] Everett, Michael, Yu Fan Chen, and Jonathan P. How. "Motion planning among dynamic, decision-making agents with deep reinforcement learning." 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018.
* [4] Chen, Changan, et al. "Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning." 2019 international conference on robotics and automation (ICRA). IEEE, 2019.

<!-- 2024-05-12 21:58:43, INFO: Using device: cpu
2024-05-12 21:58:43, INFO: Policy: LSTM-RL w/o pairwise interaction module
2024-05-12 21:58:43, INFO: human number: 5
2024-05-12 21:58:43, INFO: Not randomize human's radius and preferred speed
2024-05-12 21:58:43, INFO: Training simulation: circle_crossing, test simulation: square_crossing
2024-05-12 21:58:43, INFO: Square width: 10.0, circle width: 4.0
None
2024-05-12 21:58:43, INFO: Agent is invisible and has holonomic kinematic constraint
2024-05-12 22:45:24, INFO: TEST  has success rate: 0.92, collision rate: 0.05, nav time: 11.44, total reward: 0.2382
2024-05-12 22:45:24, INFO: Frequency of being in danger: 0.15 and average min separate distance in danger: 0.10
2024-05-12 22:45:24, INFO: Collision cases: 39 48 71 79 113 120 175 210 218 220 236 238 247 258 266 276 287 295 318 358 361 367 404 493
2024-05-12 22:45:24, INFO: Timeout cases: 2 59 161 167 176 207 228 250 364 382 408 424 492 498 -->

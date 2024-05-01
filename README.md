# Portfolio-Optimization-RL
18-786 Course Project

This project explores combining a deep reinforcement learning framework with barrier-function based risk controllers for robust portfolio management.
1. Integrated YahooFinance API with FinRL, and used FinRL for preprocessing
2. Designed a market trading environment using OpenAI Gym
3. Designed and tuned 5 RL agents to interact with the environment, using Stablebaselines3
4. Implemented a SOCP formulation of the barrier-function-based risk controller, using CVXPY
5. Reported backtesting metrics using Pyfolio

Based off:
1. **FinRL Design**: For more details, refer to the paper available at [FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance](https://arxiv.org/abs/2111.09395).
2. **Risk Controller**: For more details, refer to the paper available at [Combining Reinforcement Learning and Barrier Functions for Adaptive Risk Management in Portfolio Optimization](https://arxiv.org/pdf/2306.07013).

Our project video can be found [here](https://arxiv.org/abs/2111.09395).
The models, model result files, plots, and backtest statistics for each experiment performed can be found [here](https://docs.google.com/spreadsheets/d/1_DgW1Ay-nlSGrObAt36V2PDHxxkHzjGv2gQQa5tzg_I/edit?usp=sharing).

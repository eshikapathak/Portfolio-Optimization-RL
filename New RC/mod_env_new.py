from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from risk_controller import RiskController

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()
        # Parameters for RiskController might include
        self.risk_params = {
            "N": stock_dim,
            "sigma_s_min": 0.01, #0.01,  # Example value
            "sigma_s_max": 1, #0.02,  # Example value
            "mu": 2,           # Example value
            "rf": 0.0016,          # Risk-free rate
            "eta": 0.3,         # Example value
            "m": 0.001,            # Minimal impact factor- The higher m represents more strict risk management 
            "v": 1,           # Risk appetite factor- lower v: has less tolerance on investment risk
            "market_risk_sigma": 0.001  # Example market risk
        }

        self.risk_controller = RiskController(**self.risk_params)
        self._calculate_daily_returns()

        self.daily_returns_memory = pd.DataFrame(index=df.index, columns=df['tic'].unique())
        

    def _calculate_daily_returns(self):
        """Calculate daily returns based on close prices for each stock."""
        prices = self.df.loc[:, self.tech_indicator_list]  # Assuming close prices are among tech indicators
        returns = prices.pct_change().fillna(0)
        self.df['daily_returns'] = returns.mean(axis=1)  # Storing average daily returns across all stocks
    # def _calculate_daily_returns(self):
    #     """Calculate daily returns based on close prices for each stock and store them individually."""
    #     prices = self.df.loc[:, self.tech_indicator_list]
    #     returns = prices.pct_change().fillna(0)
    #     # Add prefix to distinguish daily return columns
    #     for col in returns.columns:
    #         self.df[f'daily_return_{col}'] = returns[col]

    def _calculate_daily_returns_stock(self):
        """Calculates daily returns for each stock and updates the daily_returns_memory."""
        #print(self.df)
        # Assuming 'close' prices are stored in the dataframe
        current_prices = self.df.loc[self.day, 'close']
        previous_prices = self.df.loc[self.day - 1, 'close'] if self.day > 0 else current_prices

        #print(previous_prices)


    #     # Calculate returns for each stock
    #     daily_returns = (current_prices - previous_prices) / previous_prices
    #     self.daily_returns_memory.loc[self.df.index[self.day]] = daily_returns
    #     print(self.daily_returns_memory)
    # def _calculate_daily_returns_stock(self):
    #     """Calculates daily returns for each stock and updates the daily_returns_memory."""
    #     # If it's the first day, skip the calculation, as there is no previous day to compare to
    #     if self.day == 0 or self.day == 1:
    #         self.daily_returns_memory.loc[self.df.index.unique()[self.day], 'daily_return'] = 0
    #     else:
    #         # Get current and previous prices
    #         current_prices = self.df.loc[self.df.index.unique()[self.day], 'close']
    #         previous_prices = self.df.loc[self.df.index.unique()[self.day - 1], 'close']

    #         # Calculate returns for each stock
    #         daily_returns = (current_prices - previous_prices) / previous_prices

    #         # Update the daily returns in the memory DataFrame
    #         self.daily_returns_memory.loc[self.df.index.unique()[self.day], 'daily_return'] = daily_returns

    #         # Optional: print to check values
    #         print(self.daily_returns_memory.loc[self.df.index.unique()[self.day]])
    # def _calculate_daily_returns_stock(self):
    #     """Calculates daily returns for each stock and updates the daily_returns_memory."""
    #     # If it's the first day or the second day, skip the calculation, as there is no previous day to compare to
    #     if self.day == 0 or self.day == 1:
    #         self.daily_returns_memory.at[self.df.index[self.day], 'daily_return'] = 0
    #     else:
    #         # Check if the current day's index is within the bounds of the DataFrame's unique index
    #         if self.day < len(self.df.index.unique()):
    #             current_day_index = self.df.index.unique()[self.day]
    #             previous_day_index = self.df.index.unique()[self.day - 1]

    #             # Get current and previous prices using safe indexing
    #             current_prices = self.df.loc[current_day_index, 'close']
    #             previous_prices = self.df.loc[previous_day_index, 'close']

    #             # Calculate returns for each stock
    #             daily_returns = (current_prices - previous_prices) / previous_prices

    #             # Safely update the daily returns in the memory DataFrame
    #             if isinstance(daily_returns, pd.Series):
    #                 # Ensure there's an existing entry for today's index; if not, create it
    #                 if current_day_index not in self.daily_returns_memory.index:
    #                     self.daily_returns_memory = self.daily_returns_memory.reindex(
    #                         self.daily_returns_memory.index.union([current_day_index]), fill_value=0)
    #                 self.daily_returns_memory.loc[current_day_index, 'daily_return'] = daily_returns
    #             else:
    #                 # For a single value, use 'at' for fast and safe scalar setting
    #                 self.daily_returns_memory.at[current_day_index, 'daily_return'] = daily_returns

    #             # Optional: print to check values
    #             print(self.daily_returns_memory.loc[current_day_index])
    #         else:
    #             print("Day index is out of bounds of the unique indices available in the DataFrame.")




    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def calculate_portfolio_weights(self):
        # Extract cash balance
        cash_balance = self.state[0]
        
        # Extract stock prices and number of shares
        stock_prices = np.array(self.state[1:1 + self.stock_dim])
        num_shares = np.array(self.state[1 + self.stock_dim:1 + 2 * self.stock_dim])
        
        # Calculate total stock value
        stock_values = stock_prices * num_shares
        total_stock_value = np.sum(stock_values)
        
        # Calculate total portfolio value
        total_portfolio_value = cash_balance + total_stock_value
        
        # Calculate weights of each stock
        stock_weights = stock_values / total_portfolio_value
        
        return stock_weights

    def step(self, actions):

    
        # Continue with risk parameters and action adjustment
        sigma_s_min = self.risk_params['sigma_s_min']

        # Extract current prices from the state
        current_prices = self.state[1:1+self.stock_dim]  # current prices for stocks

        # Initialize an empty dictionary to store expected returns for each stock
        expected_returns = {}
        returns_data = pd.DataFrame()

        # Loop through each unique stock identifier
        for stock in self.df['tic'].unique():
            # Filter data for the current stock
            stock_data = self.df[self.df['tic'] == stock]
            
            # Sort by date to ensure correct order of days
            stock_data = stock_data.sort_values('date')
            
            # Calculate the mean of the daily returns for the last three available days
            # Here you use iloc to get the last three days' returns
            if len(stock_data) >= 3:
                last_three_returns = stock_data['daily_returns'].iloc[-3:]
                expected_returns[stock] = last_three_returns.mean()
            else:
                # If less than three days of data is available, handle appropriately
                expected_returns[stock] = 0  # or use NaN or any suitable default/fallback value
            
            # Append this data to the returns_data DataFrame
            returns_data[stock] = last_three_returns.reset_index(drop=True)

        # Extracting the expected returns from the dictionary to an array
        expected_returns_array = np.array(list(expected_returns.values()))
        #print("Expected Returns Array:", expected_returns_array)

        expected_returns = expected_returns_array

        # Get recent performance from the asset memory
        #recent_performance = self.asset_memory[-10:]  # last 10 performances
        a = self.asset_memory[-10:]
        percentage_changes = [(a[i] - a[i - 1]) / a[i - 1] for i in range(1, len(a))]
        recent_performance = np.array(percentage_changes)

        #print(recent_performance)

        actions_original = actions  # Store original actions for comparison
        #print("action sum", np.sum(actions))

        # Calculate current portfolio weights
        stock_prices = np.array(current_prices)
        num_shares = np.array(self.state[1 + self.stock_dim:1 + 2 * self.stock_dim])
        stock_values = stock_prices * num_shares
        total_portfolio_value = self.state[0] + np.sum(stock_values)
        current_weights = stock_values / total_portfolio_value

        # # Update daily returns at the start of each step
        # self._calculate_daily_returns_stock()
        # if len(self.daily_returns_memory.dropna()) > 10:  # Ensure there's enough data
        #     recent_returns = self.daily_returns_memory.tail(10)  # last 30 days returns
        #     sigma_k_t1 = np.cov(recent_returns, rowvar=False)
        # else:
        #     sigma_k_t1 = np.ones((self.stock_dim, self.stock_dim))  # Default or zero matrix
        self._calculate_daily_returns_stock()

        # # Ensure there's enough data
        # if len(self.daily_returns_memory.dropna()) > 10:
        #     recent_returns = self.daily_returns_memory.tail(10).mean()  # last 10 days returns
        #     print(recent_returns.shape)
        #     sigma_k_t1 = np.cov(recent_returns.T, bias=True)  # Calculate covariance matrix
        # else:
        #     sigma_k_t1 = np.eye(self.stock_dim)  # Default to identity matrix if not enough data
        # print("expt returns", expected_returns)
        # print(expected_returns.shape, expected_returns.reshape(1, -1).shape)
        # Using rowvar=False because columns represent stocks (variables) and rows represent observations
        sigma_k_t1 = np.cov(returns_data, rowvar=False)
        if np.isnan(sigma_k_t1).any() or np.isinf(sigma_k_t1).any():
            print("NaNs or Infs found in the matrix. Applying np.nan_to_num.")
            sigma_k_t1= np.nan_to_num(sigma_k_t1)
        else:
            print("No NaNs or Infs found in the matrix. No conversion applied.")
            
        #

        # Replace NaNs in the covariance matrix with zeros or another appropriate value
        #sigma_k_t1 = np.nan_to_num(sigma_k_t1)
        # if len(expected_returns) < self.stock_dim:
        #     sigma_k_t1 = np.eye(self.stock_dim)
        # else:
        #     sigma_k_t1 = np.cov(expected_returns.reshape(-1, 1), rowvar = True)
        #     sigma_k_t1[np.isnan(sigma_k_t1)] = 1
        # print(sigma_k_t1)
        #print(recent_performance.shape, np.cov(recent_performance).shape, sigma_k_t1.shape, current_weights.shape)

        #print("good till here")
        # Adjust actions using the RiskController
        adjusted_weights = self.risk_controller.adjust_actions(
            a_rl=current_weights,
            delta_p_t1=expected_returns,  # this is an array if multiple stocks
            sigma_k_t1=sigma_k_t1,  
            sigma_alpha_t=np.std(recent_performance),  # simplistic strategy risk
            sigma_s_t=sigma_s_min,  # example static risk setting
            recent_performance=recent_performance
        )

        # Convert adjusted weights back into actions
        new_stock_values = adjusted_weights * total_portfolio_value
        new_num_shares = new_stock_values / stock_prices
        action_diffs = new_num_shares - num_shares  # Difference in shares

        #print("action_diffs", action_diffs)

        # Convert share differences to scaled actions [-1, 1]
        actions = np.clip(action_diffs / self.hmax, -1, 1)
        
        #print("action diff:", actions_original - actions)
        
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares

            #print("integer actions", actions)
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
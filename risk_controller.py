import numpy as np
import cvxpy as cp

class RiskController:
    """
    RiskController uses Barrier Function-based Risk Constraints and Adaptive Risk Strategy
    to adjust the actions from an RL agent within the portfolio optimization context.
    """

    def __init__(self, N, sigma_s_min, sigma_s_max, mu, rf, eta, m, v, market_risk_sigma):
        """
        Initializes the RiskController with specific parameters for risk management.
        
        Parameters:
        N (int): Number of assets in the portfolio
        sigma_s_min (float): Minimum acceptable risk
        sigma_s_max (float): Maximum acceptable risk
        mu (float): User-defined factor representing investor's aversion to future risk
        rf (float): Risk-free rate
        eta (float): Class K infinity function parameter
        m (float): Minimal impact of risk controllers to the overall trading strategy
        v (float): Investor's risk appetite
        market_risk_sigma (float): Market risk
        """
        self.N = N
        self.sigma_s_min = sigma_s_min
        self.sigma_s_max = sigma_s_max
        self.mu = mu
        self.rf = rf
        self.eta = eta
        self.m = m
        self.v = v
        self.market_risk_sigma = market_risk_sigma

    def calculate_lambda_t(self, recent_performance):
        """
        Calculates the scaling factor λ_t based on the recent performance of the trading strategy.
        
        Parameters:
        recent_performance (list[float]): Moving average of daily returns
        
        Returns:
        float: The scaling factor λ_t
        """
        Rs = np.mean(recent_performance)  # Moving average of daily returns
        G = min(abs(Rs - self.rf) / np.sqrt(self.v), 1)
        
        if Rs < self.rf:
            lambda_t = (self.m + G) / (1 - G)
        else:
            lambda_t = self.m
        
        return lambda_t

    def calculate_sigma_s_t1(self, expected_return):
        """
        Calculates the upper boundary of acceptable risk for the next period based on expected return.
        
        Parameters:
        expected_return (float): Expected return for the next period
        
        Returns:
        float: Upper boundary of acceptable risk for the next period
        """
        if expected_return <= (1 - self.mu) * self.rf:
            return self.sigma_s_min
        elif expected_return >= (1 + self.mu) * self.rf:
            return self.sigma_s_max
        else:
            M = (self.sigma_s_max - self.sigma_s_min) / (2 * self.mu * self.rf)
            b = ((1 + self.mu) * self.sigma_s_min - (1 - self.mu) * self.sigma_s_max) / (2 * self.mu)
            return M * expected_return + b

    def adjust_actions(self, a_rl, delta_p_t1, sigma_k_t1, sigma_alpha_t, sigma_s_t, recent_performance):
        """
        Adjusts the actions proposed by the RL agent based on the calculated risk boundary and scaling factor.
        
        Parameters:
        a_rl (np.ndarray): Actions proposed by the RL agent
        delta_p_t1 (np.ndarray): Estimated price changes for each asset
        sigma_k_t1 (np.ndarray): Covariance matrix of asset returns at t+1
        sigma_alpha_t (float): Strategy risk at time t
        sigma_s_t (float): Acceptable risk at time t
        recent_performance (list[float]): Recent performance of the trading strategy
        
        Returns:
        np.ndarray: The adjusted actions after applying risk control
        """
        # Calculate expected return and the risk upper boundary for the next period
        expected_return = np.mean(delta_p_t1)
        sigma_s_t1 = self.calculate_sigma_s_t1(expected_return)
        
        # Calculate the scaling factor based on the recent performance
        lambda_t = self.calculate_lambda_t(recent_performance)
        
        # Optimization problem variables
        a_ctrl_t = cp.Variable(self.N)
        
        # Define the SOCP constraints
        constraints = [
            # Risk constraint using the barrier function
            cp.quad_form(a_ctrl_t, sigma_k_t1) <= sigma_s_t1 - self.market_risk_sigma + 
                                                   (self.eta - 1) * (sigma_s_t - sigma_alpha_t - self.market_risk_sigma),
            
            # Control actions must be within [0, 1] after combining with RL agent's actions
            a_rl + a_ctrl_t <= 1,
            a_rl + a_ctrl_t >= 0,
            
            # The sum of the RL and control actions should equal 1
            cp.sum(a_rl + a_ctrl_t) == 1,
        ]

        # Define the optimization problem and solve
        objective = cp.Minimize(-cp.sum(a_ctrl_t))
        prob = cp.Problem(objective, constraints)
        # prob.solve()

        # # # Check if the problem is solved successfully
        # # # if prob.status not in ["infeasible", "unbounded"]:
        # # # Check if the solution is successful and valid
        # if prob.status in ["optimal", "optimal_inaccurate"] and a_ctrl_t.value is not None:
        #     # Apply the scaling factor to the optimized control actions
        #     adjusted_action = a_rl + lambda_t * a_ctrl_t.value
        #     return adjusted_action
        # else:
        #     print("Optimization problem was not solved. Returning original RL actions.")
        #     return a_rl
        # # # Try different solvers in order of preference
        # # solvers = [cp.ECOS, cp.SCS, cp.OSQP]
        # # for solver in solvers:
        # #     prob = cp.Problem(objective, constraints)
        # #     result = prob.solve(solver=solver)
        # #     if prob.status in ["optimal", "optimal_inaccurate"] and a_ctrl_t.value is not None:
        # #         adjusted_action = a_rl + lambda_t * a_ctrl_t.value
        # #         return adjusted_action
        
        # # print(f"All solvers failed; defaulting to original actions. Last solver status: {prob.status}")
        # # return a_rl  # Fallback if all solvers fail
        # Try solving with a timeout and error handling
        try:
            # Timeout set for solver in seconds (e.g., 10 seconds)
            #result = prob.solve(solver=cp.ECOS, max_iters=100, feastol=1e-4, abstol=1e-4, reltol=1e-4, verbose=False, time_limit=10)
            prob.solve(solver=cp.ECOS)
            if prob.status in ["optimal", "optimal_inaccurate"] and a_ctrl_t.value is not None:
                adjusted_action = a_rl + lambda_t * a_ctrl_t.value
                return adjusted_action
        except (cp.SolverError, Exception) as e:
            print(f"Solver encountered an issue: {str(e)}. Defaulting to original actions.")
        
        # Return unadjusted actions if there's an error or timeout
        return a_rl

# # Example usage:
# # Initialize the risk controller with parameters
# risk_controller = RiskController(N=10, sigma_s_min=0.1, sigma_s_max=0.2, mu=0.5, rf=0.02, eta=0.01, m=0.5, v=0.1, market_risk_sigma=0.15)
# # Assume some dummy values for action proposals and other parameters
# a_rl = np.array([0.9] * 10)  # Proposed actions by the RL agent
# delta_p_t1 = np.array([0.01] * 10)  # Estimated price changes
# sigma_k_t1 = np.eye(10)  # Covariance matrix
# sigma_alpha_t = 0.1  # Strategy risk
# sigma_s_t = 0.15  # Acceptable risk
# recent_performance = [0.01, 0.02, -0.01, 0.015]  # Recent performance (daily returns)

# # Adjust actions
# adjusted_actions = risk_controller.adjust_actions(a_rl, delta_p_t1, sigma_k_t1, sigma_alpha_t, sigma_s_t, recent_performance)
# print(adjusted_actions)



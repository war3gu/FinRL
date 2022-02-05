


import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
from copy import deepcopy

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import random

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
#from stable_baselines3.common import logger
import time

from stable_baselines3.common import utils
logger = utils.configure_logger()


class StockTradingEnvV2(gym.Env):

    """
    A stock trading environment for OpenAI gym
    Parameters:
    state space: {start_cash, <owned_shares>, for s in stocks{<stock.values>}, }
        df (pandas.DataFrame): Dataframe containing data
        transaction_cost (float): cost for buying or selling shares
        hmax (int): max number of share purchases allowed per asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
        reward_scaling (float): Scaling value to multiply reward by at each step.
        initial_amount: (int, float): Amount of cash initially available
        daily_information_columns (list(str)): Columns to use when building state space from the dataframe.
        out_of_cash_penalty (int, float): Penalty to apply if the algorithm runs out of cash

    action space: <share_dollar_purchases>

    TODO:
        make cash penalty proportional to total assets


    tests:
        after reset, static strategy should result in same metrics

        buy zero should result in no costs, no assets purchased
        given no change in prices, no change in asset values
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df,
            transaction_cost_pct=3e-3,
            date_col_name="date",
            hmax=10,
            turbulence_threshold=None,
            print_verbosity=10,
            reward_scaling=1e-4,
            initial_amount=1e6,
            daily_information_cols=["open", "close", "high", "low", "volume"],
            out_of_cash_penalty=None,
            cache_indicator_data=True,
            daily_reward=None,
            cash_penalty_proportion=0.1,
            random_start=True,
    ):
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start

        self.df = self.df.set_index(date_col_name)
        self.hmax = hmax
        self.initial_amount = initial_amount
        if out_of_cash_penalty is None:
            out_of_cash_penalty = -initial_amount * 0.5
        self.out_of_cash_penalty = out_of_cash_penalty
        self.print_verbosity = print_verbosity
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.daily_information_cols = daily_information_cols
        self.state_space = (
                1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.episode = -1  # initialize so we can call reset
        #self.episode_history = []
        self.printed_header = False
        self.daily_reward = daily_reward
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.cash_penalty_proportion = cash_penalty_proportion
        if self.cache_indicator_data:
            print("caching data")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            print("data cached!")

        #self.close_index = self.get_close_index()

    def seed(self, seed=None):
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    @property
    def current_step(self):
        return self.date_index - self.starting_point

    def reset(self):
        self.seed()
        self.sum_trades = 0
        if self.random_start:
            starting_point = random.choice(range(int(len(self.dates) * 0.5)))
            self.starting_point = starting_point
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": [],
        }



        #应该是记录cash， assets， date， 就可以得到一切信息，包括状态
        self.raw_state_memory = []

        self.add_raw_state(self.initial_amount, [0] * len(self.assets), self.date_index)

        init_state = self.raw_state_cast_state(-1)
        '''
        init_state = np.array(
            [self.initial_amount]
            + [0] * len(self.assets)
            + self.get_date_vector(self.date_index)
        )
        '''
        self.state_memory.append(init_state)  #account_information是多余的，根据state是可以算出来account_information的
        return init_state

    def add_raw_state(self, cash, assets, date_index):
        raw_state = {}
        raw_state["cash"] = cash
        raw_state["assets"] = assets
        raw_state["date_index"] = date_index

        self.raw_state_memory.append(raw_state)

    def raw_state_cast_state(self, index):
        raw_state = self.raw_state_memory[index]
        cash = raw_state["cash"]
        assets = raw_state["assets"]
        date_index = raw_state["date_index"]

        state = np.array(
            [cash]
            + assets
            + self.get_date_vector(date_index)
        )
        return state

    def raw_state_cast_information(self, index):
        raw_state = self.raw_state_memory[index]
        cash = raw_state["cash"]
        assets = raw_state["assets"]
        date_index = raw_state["date_index"]

        assert min(assets) >= 0
        closings = np.array(self.get_date_vector(date_index, cols=["close"]))
        asset_value = np.dot(assets, closings)
        total_value = cash + asset_value
        return cash, assets, asset_value, total_value

    def get_date_vector(self, date, cols=None):
        #print("xxxxxxx")
        #print(date)
        if (cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[date]
            v = []
            for a in self.assets:
                subset = trunc_df[trunc_df[self.stock_col] == a]
                #print(subset["close"])
                ddd = subset.loc[date, cols]
                v += ddd.tolist()
            assert len(v) == len(self.assets) * len(cols)
            return v

    def return_terminal(self, reason="Last Date", reward=0):

        state = self.raw_state_cast_state(-1)
        self.log_step(reason=reason, terminal_reward=reward)
        reward = reward * self.reward_scaling
        # Add outputs to logger interface
        '''
        reward_pct = self.account_information["total_assets"][-1] / self.initial_amount
        logger.record("environment/total_reward_pct", (reward_pct - 1) * 100)
        logger.record(
            "environment/daily_trades",
            self.sum_trades / (self.current_step) / len(self.assets),
            )
        logger.record("environment/completed_steps", self.current_step)
        logger.record(
            "environment/sum_rewards", np.sum(self.account_information["reward"])
        )
        logger.record(
            "environment/cash_proportion",
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1],
            )
        '''
        return state, reward, True, {}

    def log_step(self, reason, terminal_reward=None):
        cash, assets, asset_value, total_value = self.raw_state_cast_information(-1)

        if terminal_reward is None:
            terminal_reward = self.get_reward()
        cash_pct = cash/total_value
        rec = [
            self.episode,
            self.current_step,
            reason,
            total_value,
            terminal_reward,
            cash_pct
            ]

        #self.episode_history.append(rec)
        print(self.template.format(*rec))

    def log_header(self):
        self.template = "{0:4}|{1:4}|{2:15}|{3:10}|{4:10}|{5:10}"  # column widths: 8, 10, 15, 7, 10
        print(
            self.template.format(
                "EPISODE",
                "STEPS",
                "TERMINAL_REASON",
                "TOT_ASSETS",
                "TERMINAL_REWARD_unsc",
                "CASH_PCT",
            )
        )
        self.printed_header = True

    #需要抽象一个函数，直接根据状态得到total assets,感觉记录total assets是多余的，除了让程序快一点
    def get_reward(self):                  #seems wrong， reward is (cash + assets) - (cash_last_step + assets_last_step)
        if self.current_step==0:
            return 0
        else:
            cash_last, assets_last, asset_value_last, total_value_last = self.raw_state_cast_information(-2)
            cash, assets, asset_value, total_value = self.raw_state_cast_information(-1)

            total_value_last = self.clip_total_assets(total_value_last, cash_last)
            total_value = self.clip_total_assets(total_value, cash)
            reward = (total_value - total_value_last)/self.initial_amount
            return reward

    def clip_total_assets(self, total_assets, cash):
        cash_penalty = max(0, (total_assets*self.cash_penalty_proportion-cash))   #punish for too little cash
        total_assets -= cash_penalty
        return total_assets

    def step(self, actions):
        # let's just log what we're doing in terms of max actions at each step.
        self.sum_trades += np.sum(np.abs(actions))

        # print header only first time
        if self.printed_header is False:
            self.log_header()

        # print if it's time.
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")

        # if we're at the end
        if self.date_index == len(self.dates) - 1:
            # if we hit the end, set reward to total gains (or losses)
            return self.return_terminal(reward=0)                    #最后一天，没有操作，reward为0
        else:
            # compute value of cash + assets

            cash, assets, asset_value, total_value = self.raw_state_cast_information(-1)

            closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))



            # log the values of cash, assets, and total assets

            self.account_information["cash"].append(cash)                         #record cash
            self.account_information["asset_value"].append(asset_value)                 #record asset
            self.account_information["total_assets"].append(total_value)   #record total asset
            self.account_information["reward"].append(0)                           #record reward


            # multiply action values by our scalar multiplier and save
            actions = actions * self.hmax                     #money
            self.actions_memory.append(actions)

            # scale cash purchases to asset # changes
            actions = actions / closings                      #stock count change
            self.transaction_memory.append(actions)

            # clip actions so we can't sell more assets than we hold
            actions = np.maximum(actions, -np.array(assets))  #-表示卖出，不能卖空

            # compute our proceeds from sales, and add to cash
            sells = -np.clip(actions, -np.inf, 0)             #stock sell
            proceeds = np.dot(sells, closings)                #gain cash
            costs = proceeds * self.transaction_cost_pct      #cost cash
            coh = cash + proceeds                             #current cash

            # compute the cost of our buys
            buys = np.clip(actions, 0, np.inf)                 #stock buy
            spend = np.dot(buys, closings)                     #spend cash
            costs += spend * self.transaction_cost_pct         #cost cash

            # if we run out of cash, end the cycle and penalize
            if (spend + costs) > coh:                          #买+交易费>卖+当前金钱
                return self.return_terminal(                   #没有操作，reward为0
                    reason="CASH SHORTAGE",
                    reward=0
                )

            # verify we didn't do anything impossible here
            assert (spend + costs) <= coh

            # update our holdings
            coh = coh - spend - costs                         #update total asset
            holdings_updated = assets + actions               #update hlodings
            self.date_index += 1                              #new day


            #self.state_memory.append(state)
            self.add_raw_state(coh, list(holdings_updated), self.date_index)  #新状态，新一天

            #此时才能计算reward。操作后第二天的资产减去操作前第一天的资产

            # reward is (cash + assets) - (cash_last_step + assets_last_step)
            reward = self.get_reward()


            state = self.raw_state_cast_state(-1)

            reward = reward * self.reward_scaling
            return state, reward, False, {}



    def get_sb_env(self):
        def get_self():
            return deepcopy(self)

        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(self, n=10):
        def get_self():
            return deepcopy(self)

        #e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        e = SubprocVecEnv([get_self for _ in range(n)], start_method=None)  #Only ‘forkserver’ and ‘spawn’ start methods are thread-safe
        #e = SubprocVecEnv([get_self for _ in range(n)])
        #e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs



    def save_asset_memory(self):
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                                               -len(self.account_information["cash"]) :
                                               ]
            
            return pd.DataFrame(self.account_information)



    def save_action_memory(self):
        if self.current_step == 0:
            return None
        else:
            return pd.DataFrame(
                {
                    "date": self.dates[-len(self.account_information["cash"]) :],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory,
                }
            )

print(StockTradingEnvV2.__doc__)
# In[ ]:
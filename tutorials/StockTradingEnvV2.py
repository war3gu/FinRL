


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
            out_of_cash_penalty=None,                                  #金额用完的惩罚比例，可以避免频繁的透支
            cache_indicator_data=True,
            cash_penalty_proportion=0.1,
            random_start=True,
    ):
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start

        self.df = self.df.set_index(date_col_name)
        self.hmax = hmax                                      #每只股票操作的最大金额
        self.initial_amount = initial_amount

        if not out_of_cash_penalty:
            out_of_cash_penalty = 0.001
        self.out_of_cash_penalty = out_of_cash_penalty         #超支之后，对cash的惩罚,同时没有没有动作

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

        self.episode += 1

        if self.random_start:
            starting_point = random.choice(range(int(len(self.dates) * 0.1)))
            self.starting_point = starting_point
        else:
            self.starting_point = 0

        self.date_index = self.starting_point

        self.raw_state_memory = []    #应该是记录cash， assets， date， 就可以得到一切信息，包括状态

        self.add_raw_state(np.array([0] * len(self.assets)), np.array([0] * len(self.assets)), self.initial_amount, [0] * len(self.assets), self.date_index)

        init_state = self.raw_state_cast_state(-1)

        return init_state

    def add_raw_state(self,cash_change, asset_change, cash, assets, date_index): #cash_change,asset_change都是没clip的
        raw_state = {}
        raw_state["cash_change"] = cash_change
        raw_state["asset_change"] = asset_change
        raw_state["cash"] = cash
        raw_state["assets"] = assets
        raw_state["date_index"] = date_index

        self.raw_state_memory.append(raw_state)

    def get_raw_state(self, index):
        memory = self.raw_state_memory[index]
        return memory["cash_change"], memory["asset_change"], memory["cash"], memory["assets"], memory["date_index"]

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

        date = self.dates[date_index]

        assert min(assets) >= 0
        closings = np.array(self.get_date_vector(date_index, cols=["close"]))
        asset_value = np.dot(assets, closings)
        total_value = cash + asset_value
        return cash, assets, asset_value, total_value, date

    def raw_state_cast_action_memory(self):
        raw_state_memory = pd.DataFrame(self.raw_state_memory)
        date_index = raw_state_memory["date_index"]
        date = self.dates[date_index]
        cash_change = raw_state_memory["cash_change"]
        asset_change= raw_state_memory["asset_change"]

        df = pd.DataFrame(
            {
                "date": date,
                "actions": cash_change,                          #cash   change
                "transactions": asset_change,                    #asset  changes
            }
        )

        return df[1:]                                            #第一天是没操作的

    def raw_state_cast_information_memory(self):
        cash_list = []
        asset_value_list = []
        total_value_list = []
        date_list = []
        reward_list = []
        length = len(self.raw_state_memory)
        for i in range(1, length):
            cash, _, asset_value, total_value, date = self.raw_state_cast_information(i)
            reward = self.get_reward_by_index(i, i-1)
            cash_list.append(cash)
            asset_value_list.append(asset_value)
            total_value_list.append(total_value)
            date_list.append(date)
            reward_list.append(reward)

        df = pd.DataFrame(
            {
                "date": date_list,
                "cash": cash_list,
                "asset_value": asset_value_list,
                "total_assets": total_value_list,
                "reward": reward_list
            }
        )

        return df

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

        return state, reward, True, {}

    def log_step(self, reason, terminal_reward=None):
        cash, assets, asset_value, total_value, _ = self.raw_state_cast_information(-1)

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
            return self.get_reward_by_index(-1, -2)

    def get_reward_by_index(self, index, index_last):
        cash_last, assets_last, asset_value_last, total_value_last, _ = self.raw_state_cast_information(index_last)
        cash, assets, asset_value, total_value, _ = self.raw_state_cast_information(index)

        total_value_last = self.clip_total_assets(total_value_last, cash_last)
        total_value = self.clip_total_assets(total_value, cash)
        reward = (total_value - total_value_last)/self.initial_amount

        return reward

    def clip_total_assets(self, total_assets, cash):
        cash_penalty = max(0, (total_assets*self.cash_penalty_proportion-cash))   #punish for too little cash
        total_assets -= cash_penalty
        return total_assets

    def step(self, actions):

        if self.random_start == False:
            print("date_index = {0}".format(self.date_index))

        # print header only first time
        if self.printed_header is False:
            self.log_header()

        # print if it's time.
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")

        if self.date_index == len(self.dates) - 1:                          #为了和models模块契合，传进来的self.date_index == len(self.dates) - 1才能结束
            self.log_step(reason="Last Date", terminal_reward=0)

            state = self.raw_state_cast_state(-1)

            return state, 0, True, {}
        else:
            cash, assets, asset_value, total_value, _ = self.raw_state_cast_information(-1)

            closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))

            actions = actions * self.hmax                     #money
            cash_change = actions                             #金钱的变化

            actions = actions / closings                      #stock count change
            asset_change = actions                            #股票持有量的变化

            actions = np.maximum(actions, -np.array(assets))  #-表示卖出，不能卖空


            sells = -np.clip(actions, -np.inf, 0)             #stock sell
            gains = np.dot(sells, closings)                   #gain cash
            costs = gains * self.transaction_cost_pct         #cost cash
            cash_after_sell = cash + gains                    #current cash


            buys = np.clip(actions, 0, np.inf)                 #stock buy
            spend = np.dot(buys, closings)                     #spend cash
            costs += spend * self.transaction_cost_pct         #cost cash

            #test的时候碰到这个不是会崩溃吗？因为会直接退出，得不到account_memory
            if (spend + costs) > cash_after_sell:              #此处如果不结束，会出现date重复，可能导致DRL_prediction出问题。也不用add_raw_state

                penalty = -self.out_of_cash_penalty

                _, _, cash_last, assets_last, date_index_last = self.get_raw_state(-1)

                cash_last = cash_last + self.initial_amount * penalty

                state_last = np.array(
                    [cash_last]
                    + assets
                    + self.get_date_vector(date_index_last)
                )
                self.log_step(reason="CASH SHORTAGE", terminal_reward=penalty)

                if self.random_start == False:                 #这个示范代码写的不好，换个看看
                    print("this is test, crash will happen")

                return state_last, penalty, True, {}                      #重重的惩罚

            assert (spend + costs) <= cash_after_sell

            cash_after_buy = cash_after_sell - spend - costs                   #update cash
            holdings_updated = assets + actions                                #update hlodings

            self.date_index += 1                                               #new day
            self.add_raw_state(cash_change, asset_change, cash_after_buy, list(holdings_updated), self.date_index)

            reward = self.get_reward()              # reward is (cash + assets) - (cash_last_step + assets_last_step)  此时才能计算reward。操作后第二天的资产减去操作前第一天的资产

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
            return self.raw_state_cast_information_memory()



    def save_action_memory(self):
        if self.current_step == 0:
            return None
        else:
            return self.raw_state_cast_action_memory()

print(StockTradingEnvV2.__doc__)
# In[ ]:
#!/usr/bin/env python
# coding: utf-8

# # Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading
# 
# Tutorials to use OpenAI DRL to trade multiple stocks in one Jupyter Notebook | Presented at NeurIPS 2020: Deep RL Workshop
# 
# * This blog is based on our paper: FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, presented at NeurIPS 2020: Deep RL Workshop.
# * Check out medium blog for detailed explanations: https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530
# * Please report any issues to our Github: https://github.com/AI4Finance-LLC/FinRL-Library/issues
# * **Pytorch Version** 
# 
# 

# # Content

# * [1. Problem Definition](#0)
# * [2. Getting Started - Load Python packages](#1)
#     * [2.1. Install Packages](#1.1)    
#     * [2.2. Check Additional Packages](#1.2)
#     * [2.3. Import Packages](#1.3)
#     * [2.4. Create Folders](#1.4)
# * [3. Download Data](#2)
# * [4. Preprocess Data](#3)        
#     * [4.1. Technical Indicators](#3.1)
#     * [4.2. Perform Feature Engineering](#3.2)
# * [5.Build Environment](#4)  
#     * [5.1. Training & Trade Data Split](#4.1)
#     * [5.2. User-defined Environment](#4.2)   
#     * [5.3. Initialize Environment](#4.3)    
# * [6.Implement DRL Algorithms](#5)  
# * [7.Backtesting Performance](#6)  
#     * [7.1. BackTestStats](#6.1)
#     * [7.2. BackTestPlot](#6.2)   
#     * [7.3. Baseline Stats](#6.3)   
#     * [7.3. Compare to Stock Market Index](#6.4)             

# <a id='0'></a>
# # Part 1. Problem Definition

# This problem is to design an automated trading solution for single stock trading. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.
# 
# The algorithm is trained using Deep Reinforcement Learning (DRL) algorithms and the components of the reinforcement learning environment are:
# 
# 
# * Action: The action space describes the allowed actions that the agent interacts with the
# environment. Normally, a ∈ A includes three actions: a ∈ {−1, 0, 1}, where −1, 0, 1 represent
# selling, holding, and buying one stock. Also, an action can be carried upon multiple shares. We use
# an action space {−k, ..., −1, 0, 1, ..., k}, where k denotes the number of shares. For example, "Buy
# 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively
# 
# * Reward function: r(s, a, s′) is the incentive mechanism for an agent to learn a better action. The change of the portfolio value when action a is taken at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio
# values at state s′ and s, respectively
# 
# * State: The state space describes the observations that the agent receives from the environment. Just as a human trader needs to analyze various information before executing a trade, so
# our trading agent observes many different features to better learn in an interactive environment.
# 
# * Environment: Dow 30 consituents
# 
# 
# The data of the single stock that we will be using for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
# 

# <a id='1'></a>
# # Part 2. Getting Started- ASSUMES USING DOCKER, see readme for instructions

# <a id='1.1'></a>
# ## 2.1. Add FinRL to your path. You can of course install it as a pipy package, but this is for development purposes.
# 

# In[1]:


import sys

sys.path.append("..")


# In[2]:


import pandas as pd
print(pd.__version__)


# 
# <a id='1.2'></a>
# ## 2.2. Check if the additional packages needed are present, if not install them. 
# * Yahoo Finance API
# * pandas
# * numpy
# * matplotlib
# * stockstats
# * OpenAI gym
# * stable-baselines
# * tensorflow
# * pyfolio

# <a id='1.3'></a>
# ## 2.3. Import Packages

# In[5]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import datetime

#get_ipython().run_line_magic('matplotlib', 'inline')
from finrl.apps import config
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint

from StockTradingEnvV2 import *

import multiprocessing

import sys
sys.path.append("../FinRL-Library")

import torch
print('torch is_available {0}'.format(torch.cuda.is_available()))

'''
x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)
a, b = torch.broadcast_tensors(x, y)
print(a.size())
print(a)
print(b.size())
print(b)
'''

import itertools


# <a id='1.4'></a>
# ## 2.4. Create Folders

# In[6]:


import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


# <a id='2'></a>
# # Part 3. Download Data
# Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** to fetch data from Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).
# 

# 
# 
# -----
# class YahooDownloader:
#     Provides methods for retrieving daily stock data from
#     Yahoo Finance API
# 
#     Attributes
#     ----------
#         start_date : str
#             start date of the data (modified from config.py)
#         end_date : str
#             end date of the data (modified from config.py)
#         ticker_list : list
#             a list of stock tickers (modified from config.py)
# 
#     Methods
#     -------
#     fetch_data()
#         Fetches data from yahoo API
# 

# In[7]:


# from config.py start_date is a string
config.START_DATE


# In[8]:


# from config.py end_date is a string
config.END_DATE


# In[12]:


print(config.DOW_10_TICKER)


# In[ ]:

import platform

if __name__=="__main__":
    proxy = None
    total_timesteps = 500000           #总的采样数量
    ppo_params ={'n_steps': 2048,       #n_steps表示一次采样的数据长度
                 'ent_coef': 0.01,
                 'learning_rate': 0.00009,
                 'batch_size': 512,     #gpu跑一次的数据长度
                 'gamma': 0.99,
                 'n_epochs': 100}      #一次采样的数据使用多少次,这个足够大才能看到gpu的使用率的明显提升
    policy_kwargs = {"net_arch": [1024, 1024,1024, 1024,  1024]}
    start_date = '2009-01-01'
    mid_date = '2020-01-01'
    end_date = '2021-10-31'


    if platform.system() == 'Windows':
        proxy = '127.0.0.1:10808'
        total_timesteps = 1000
        ppo_params ={'n_steps': 256,
                     'ent_coef': 0.01,
                     'learning_rate': 0.00009,
                     'batch_size': 64,
                     'gamma': 0.99,
                     'n_epochs': 10}
        policy_kwargs = {"net_arch": [1024, 1024, 1024]}
        start_date = '2019-01-01'
        mid_date   = '2020-01-01'
        end_date   = '2021-01-01'



    df = YahooDownloader(start_date=start_date,
                         end_date=end_date,
                         ticker_list=config.DOW_10_TICKER).fetch_data(proxy=proxy)
    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                         use_turbulence=True,
                         user_defined_feature=False)

    print(df.head())

    processed = fe.preprocess_data(df)
    processed['log_volume'] = np.log(processed.volume*processed.close)
    processed['change'] = (processed.close-processed.open)/processed.close
    processed['daily_variance'] = (processed.high-processed.low)/processed.close

    print(processed.head())

    train = data_split(processed, start_date, mid_date)
    trade = data_split(processed, mid_date, end_date)
    print(train.head())
    print(trade.head())

    information_cols = ['daily_variance', 'change', 'log_volume', 'close','day',
                    'macd', 'rsi_30', 'cci_30', 'dx_30', 'turbulence']

    e_train_gym = StockTradingEnvV2(df=train,
                                    initial_amount=1e6,
                                    hmax=5000,
                                    out_of_cash_penalty=0,
                                    cache_indicator_data=False,
                                    cash_penalty_proportion=0.2,
                                    reward_scaling=1,
                                    daily_information_cols=information_cols,
                                    print_verbosity=500,
                                    random_start=True)

    e_trade_gym = StockTradingEnvV2(df=trade,
                                    initial_amount=1e6,
                                    hmax=5000,
                                    out_of_cash_penalty=0,
                                    cash_penalty_proportion=0.2,
                                    reward_scaling=1,
                                    cache_indicator_data=False,
                                    daily_information_cols=information_cols,
                                    print_verbosity=500,
                                    random_start=False)


# ## Environment for Training
# There are two available environments. The multiprocessing and the single processing env. 
# Some models won't work with multiprocessing. 
# 
# ```python
# # single processing
# env_train, _ = e_train_gym.get_sb_env()
# 
# 
# #multiprocessing
# env_train, _ = e_train_gym.get_multiproc_env(n = <n_cores>)
# ```
# 

# In[ ]:


# for this example, let's do multiprocessing with n_cores-2



    n_cores = multiprocessing.cpu_count()
    #n_cores = 24
    #n_cores = 1
    #env_train, _ = e_train_gym.get_multiproc_env(n = n_cores)

    env_train, _ = e_train_gym.get_sb_env()
    env_trade, _ = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)



    model = agent.get_model("ppo",
                            model_kwargs=ppo_params,
                            policy_kwargs=policy_kwargs,
                            verbose=1)
    model.learn(total_timesteps=total_timesteps,    #5000000   采样的总数量
                eval_env=env_trade,
                eval_log_path='.',
                eval_freq=250,
                log_interval=1,
                tb_log_name='1_18_lastrun',
                n_eval_episodes=1)

    model.save("different.model")
    data_turbulence = processed[(processed.date<mid_date) & (processed.date>=start_date)]
    insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])
    insample_turbulence.turbulence.describe()
    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values,1)
    trade.head()

    e_trade_gym.hmax = 5000
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model,environment = e_trade_gym)
    df_account_value.head(50)

    print("==============Get Backtest Results===========")

    perf_stats_all = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')

    print("==============Compare to DJIA===========")

    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(df_account_value,
                  baseline_ticker='^DJI',
                  baseline_start=mid_date,
                  baseline_end=end_date,
                  value_col_name='total_assets')

    print('end process')





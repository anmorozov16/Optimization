import pandas as pd
from keras.layers import Flatten, Input, Dense, concatenate
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import PrioritizedMemory
from rl.processors import MultiInputProcessor
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split

import Opt
import utils


NUM_OF_USER_INPUTS = 37 - 8
df = pd.read_excel('dataset/design_recom.xlsx', na_values=[""], decimal=',')
mean = df.iloc[:, NUM_OF_USER_INPUTS:-1].mean(axis=0)
std = df.iloc[:, NUM_OF_USER_INPUTS:-1].std(axis=0)

scaler = StandardScaler()

X = df.drop(['oil_production12'], axis=1)
y = df['oil_production12']

# X_scaled = scaler.fit_transform(df.drop(['oil_production12'], axis=1))
# X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, random_state=420)

model_y = CatBoostRegressor(l2_leaf_reg=0.4, silent=True, random_state=420).fit(X, y)

print(df.shape)
len_steps = df.iloc[:, NUM_OF_USER_INPUTS:-1]
len_steps = (len_steps.max(axis=0) - len_steps.min(axis=0)) / 10

# model_y = utils.ModelSimulationTarget()
train_obj = Opt.Train(df.iloc[:, :NUM_OF_USER_INPUTS], df.iloc[:, -1], model_y)
env = Opt.Recommendations([0.] * 8, train=train_obj, len_steps=len_steps, limits=[-10, 10])


frame = Input(shape=(1, NUM_OF_USER_INPUTS))
custom = Flatten()(frame)
custom = Dense(32, activation='relu')(custom)

frame1 = Input(shape=(1, 2))
custom1 = Flatten()(frame1)
custom1 = Dense(32, activation='relu')(custom1)

x = concatenate([custom, custom1])
x = Dense(16, activation='relu')(x)
buttons = Dense(3, activation='linear')(x)
model = Model(inputs=[frame, frame1], outputs=buttons)
# plot_model(model, to_file='model.png')

memory = PrioritizedMemory(limit=200000, window_length=1)

policy = EpsGreedyQPolicy(0.02)

dqn = DQNAgent(model=model, nb_actions=env.nb_actions, policy=policy, memory=memory, nb_steps_warmup=10000,
               gamma=1., target_model_update=288, processor=MultiInputProcessor(2), enable_double_dqn=True,
               train_interval=32, delta_clip=1., enable_dueling_network=True)

dqn.model.summary()
dqn.compile(Adam(lr=0.002), metrics=['mae'])
# dqn.load_weights('name_weights.h5')
# print('loaded')

dqn.fit(env, nb_steps=200000, verbose=2, visualize=False)  # время обучения ~ 10 мин.
print('save_weights')
dqn.model.save('dataset/recom_model.h5')
print('model.save')
import pandas as pd
import Opt
import utils


NUM_OF_USER_INPUTS = 37 - 8
df = pd.read_excel('dataset/design_recom.xlsx', na_values=[""], decimal=',')


print(df.shape)
len_steps = df.iloc[:, NUM_OF_USER_INPUTS:-1]
len_steps = (len_steps.max(axis=0) - len_steps.min(axis=0)) / 10
model_y = utils.ModelSimulationTarget()
env = Opt.Recommendations([0.] * 8, len_steps=len_steps, limits=[-10, 10])
print(env.predict(utils.ModelOptimization(), [1.] * NUM_OF_USER_INPUTS))

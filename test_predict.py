import pandas as pd
import Opt
from keras.models import load_model


NUM_OF_USER_INPUTS = 37 - 8
df = pd.read_excel('dataset/design_recom.xlsx', na_values=[""], decimal=',')
model = load_model('dataset/recom_model.h5', compile=False)

print(df.shape)
len_steps = df.iloc[:, NUM_OF_USER_INPUTS:-1]
len_steps = (len_steps.max(axis=0) - len_steps.min(axis=0)) / 10
env = Opt.Recommendations([0.] * 8, len_steps=len_steps, limits={'min': -10, 'max': 10})
print(env.predict(model, [1.] * NUM_OF_USER_INPUTS))

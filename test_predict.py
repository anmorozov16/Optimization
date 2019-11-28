import pandas as pd
import Opt
from keras.models import load_model


NUM_OF_USER_INPUTS = 37 - 8
df = pd.read_excel('dataset/design_recom.xlsx', na_values=[""], decimal=',')
model = load_model('dataset/recom_model.h5', compile=False)

print(df.shape)
len_steps = df.iloc[:, NUM_OF_USER_INPUTS:-1]
len_steps = (len_steps.max(axis=0) - len_steps.min(axis=0)) / 10  # можно передавать масисвом
env = Opt.Recommendations([2437, 80, 92, 1.89, 0.862, 1.003, 1.3, 33.8, 5000, 0.236, 20, 24.36, 8.15, 53, 1.16,
                           10.6, 1.89, 1.02, 4.36, 0.33, 5.4, 6.6, 0.62, 17, 1.3, 26, 26, 26, 135], len_steps=len_steps,
                          limits=[-10, 10])
print(env.predict(model, [1.] * NUM_OF_USER_INPUTS))

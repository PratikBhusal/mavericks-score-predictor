import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x = pd.read_csv('frames\\2013-10-29-CHI-MIA.csv')

sns.scatterplot(x=x['ball_x'],y=x['ball_y'],size=x['ball_z'],sizes=(2,20),hue=x['ball_z'])
plt.show()
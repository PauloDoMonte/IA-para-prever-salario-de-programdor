import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sb

db = pd.read_csv("db.csv",sep=",")

x = db.drop(columns='unfinished')
y = db.unfinished

horas = []
preco = []

h = min(db['expected_hours'])
p = min(db['price'])

h_max = max(db['expected_hours'])
p_max = max(db['price'])

incremento = 1

while(h<h_max):
    while(p<p_max):
        preco.append(p)
        horas.append(h)

        p+= incremento

    p = min(db['price'])
    h += incremento
    print(h)


df = pd.DataFrame({"expected_hours":horas,"price":preco})

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1)

clr = LogisticRegression()
clr.fit(x_train, y_train)

resultado = clr.predict(df)
df['resultado'] = resultado

'''
heatmap_data = pd.pivot(df,values='resultado',columns='expected_hours',index='price')

ax = sb.heatmap(heatmap_data, cbar_kws={'label':'Unfinished','ticks':[0,1]})
plt.title("Relação entre preço por hora de trabalho no Mercado")
plt.ylabel("Preço [$]")
plt.xlabel("Horas de trabalho")
plt.savefig("ieee.png")
'''
'''
plt.title("Preco por horas trabalhadas aceito no Mercado")
plt.scatter(df.loc[df['resultado']==1]['expected_hours'],df.loc[df['resultado']==1]['price'],color='blue',s=1)
plt.ylabel("Preço [$]")
plt.xlabel("Horas")
plt.grid()
plt.savefig("ieee2.png")'''

'''
plt.title("Preco por horas trabalhadas não aceito no Mercado")
plt.scatter(df.loc[df['resultado']==0]['expected_hours'],df.loc[df['resultado']==0]['price'],color='blue',s=1)
plt.ylabel("Preço [$]")
plt.xlabel("Horas")
plt.grid()
plt.savefig("ieee3.png")'''

print(resultado)

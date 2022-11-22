import streamlit as st
import altair as alt


# Importamos 
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
from scipy.stats import skew, kurtosis
from utils import *
import plotly.graph_objects as go




st.set_page_config(page_title="Ars Longa Capital",layout="wide", menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Ars Longa Capital \n "
    })

# Declaramos el título de la página
st.title('Ars Longa Capital')


num_comp = st.sidebar.slider('Numero de componentes', 2, 12, 2)

gen_spaces_button = st.sidebar.button("Seleccionar")
#random_button = st.sidebar.button("Generar portafolio aleatorio")

etfs = []

if gen_spaces_button:
    for i in range(num_comp):
        etf = st.sidebar.text_input(f"ETF {i+1}", key=i)
        etfs.append(etfs)
    
    #generate_button = st.sidebar.button("Generar Portafolio", key="generate_button")



# importation of data
list_tickers = ["EWC","IAU", "IVV", "QQQ", "VGK", "XLC", "XLE", "XLF", "XLK", "XLV", "EWZ"]
database = yf.download(list_tickers)


# Take only the adjusted stock price
database = database["Adj Close"]



if len(list_tickers)<6:
    # Lista vacía para guardar nombres de variables
    command = []
    for i in range(len(list_tickers)):
        command.append(f"col{i}")

    # Convertir lista en una cadena de texto
    lines = ', '.join(command)

    # ejecutar 
    exec(f"{lines} = st.columns({len(list_tickers)})")

    for i in range(len(list_tickers)):
        exec(f"col{i}.metric('{database.columns[i]}', '{np.round(database.dropna().iloc[-1,:][i], 2)}', '{np.round(database.pct_change(1).iloc[-1,:][i]*100, 3)}')")

else:
    command1 = []
    command2 = []
    for i in range(6):
        command1.append(f"col{i}")
    for i in range(6, 12):
        command2.append(f"col{i}")

    lines1 = ', '.join(command1)
    lines2 = ', '.join(command2)

    exec(f"{lines1} = st.columns({6})")
    for i in range(6):
        exec(f"col{i}.metric('{database.columns[i]}', '{np.round(database.dropna().iloc[-1,:][i], 2)}', '{np.round(database.pct_change(1).iloc[-1,:][i]*100, 3)}')")

    exec(f"{lines2} = st.columns({6})")
    for i in range(6, 11):
        exec(f"col{i}.metric('{database.columns[i]}', '{np.round(database.dropna().iloc[-1,:][i], 2)}', '{np.round(database.pct_change(1).iloc[-1,:][i]*100, 3)}')")



#st.table(database.tail(1))


# Metrics
#st.metric(label="EWC", value=f"${np.round(database.dropna().iloc[-1,:][0], 2)}", delta=f"{np.round(database.pct_change(1).iloc[-1,:][0], 3)}%")



# Crear gráfico de lineas
st.line_chart(database)

# Drop missing values
data = database.dropna().pct_change(1).dropna()




# definir train dn test sets
split = int(0.991 * len(data))
train_set = data.iloc[:split, :]
test_set = data.iloc[split:, :]
n = data.shape[1]
x0 = np.ones(n)

# Optimization constraints problem
cons = ({'type': 'eq', 'fun': lambda x: sum(abs(x)) - 1}) # we define investor to use all capital, 

# Set the bounds (from 0 to 1, because we want a long strategy, if we want to short, we set it t be [-1, 1])
Bounds = [(0, 1) for i in range(0, n)]

############################ MV CRITERION ############################ 

# Optimization problem solving (minimize the opposite to Umv)
res_MV = minimize(MV_criterion, x0, method="SLSQP", # Sequential Least SQuares Programming optimizer
                  args=(train_set), bounds=Bounds,
                  constraints=cons, options={'disp': True})

# Result for computations (extract the optimal weight)
X_MV = res_MV.x

# Compute the cumulative return of the portfolio (CM)
portfolio_return_MV = np.multiply(test_set,np.transpose(X_MV)) # multiply wieghts to test set
portfolio_return_MV = portfolio_return_MV.sum(axis=1) # sum of all returns


############################ SK CRITERION ############################ 
# Optimization problem solving
res_SK = minimize(SK_criterion, x0, method="SLSQP",
                  args=(train_set), bounds=Bounds,
                  constraints=cons, options={'disp': True})

# Result 
X_SK = res_SK.x

# Compute the cumulative return of the portfolio (CM)
portfolio_return_SK = np.multiply(test_set,np.transpose(X_SK))
portfolio_return_SK = portfolio_return_SK.sum(axis=1)




############################ SHARPE CRITERION ############################ 
# Optimization problem solving
res_SR = minimize(SR_criterion, x0, method="SLSQP",
                  args=(train_set),bounds=Bounds,
                  constraints=cons,options={'disp': True})

# Result for computations
X_SR = res_SR.x

# Compute the cumulative return of the portfolio (CM)
portfolio_return_SR = np.multiply(test_set,np.transpose(X_SR))
portfolio_return_SR = portfolio_return_SR.sum(axis=1)


############################ SORTINO CRITERION ############################ 
# Optimization problem solving
res_SOR = minimize(SOR_criterion, x0, method="SLSQP",
                  args=(train_set),bounds=Bounds,
                  constraints=cons,options={'disp': True})

# Result for computations
X_SOR = res_SOR.x


# Compute the cumulative return of the portfolio (CM)
portfolio_return_SOR = np.multiply(test_set,np.transpose(X_SOR))
portfolio_return_SOR = portfolio_return_SOR.sum(axis=1)


############################ PRINT ALL ############################ 
e = pd.concat([np.cumsum(portfolio_return_MV)*100, np.cumsum(portfolio_return_SK)*100, np.cumsum(portfolio_return_SR)* 100, np.cumsum(portfolio_return_SOR)*100], axis=1)

e = e.rename(columns={0 : "Mean Variance", 1: "Mean Var Skew Kurt", 2:"Sharpe", 3:"Sortino"})



ccol, wcol = st.columns(2)

with ccol:
    st.line_chart(e)


with wcol:
    fig = go.Figure(go.Indicator(
        mode = "number",
        value = database.iloc[-1,:].sum(),
        number = {'prefix': "$"},
        #title = {"text": "Total in USD<br><span style='font-size:0.8em;color:gray'>Subtitle</span><br><span style='font-size:0.8em;color:gray'>Subsubtitle</span>"}
        title = {"text": "Total USD to be spent<br><span style='font-size:0.8em;color:gray'>Assuming just 1 share is bought</span><br><span style='font-size:0.8em;color:gray'>"}
        ))

    st.plotly_chart(fig, use_container_width=True)


all_weights = pd.DataFrame(np.round(np.array([X_MV, X_SK, X_SR, X_SOR]), 4)*100, columns=list_tickers, index=["Mean Variance", "Mean Var Skew Kurt", "Sharpe","Sortino"])
st.bar_chart(all_weights)














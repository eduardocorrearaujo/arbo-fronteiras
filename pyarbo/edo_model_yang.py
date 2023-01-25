import numbers
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from get_data import get_weather_data 
from parameters import dict_d, dict_mu_a, dict_mu_m, dict_gamma_m, dict_theta_m

# Os parâmetros abaixo são constantes e não serão fitados, por essa razão são definidos com letra maiúscula 
MU_H = 1/(365*76)    #human mortality rate - day^-1
K = 0.5          #fraction of female mosquitoes hatched from all egs
C_A = 0.0      #control effort rate on aquatic phase
C_M = 0.0    #control effort rate on terretrial phase
THETA_H = 0.027   #intrinsic incubation rate - day^-1
ALPHA_H = 0.1 #recovering rate - day^-1
D = 4; 

def C0(Ms,k,delta,gamma_m, mu_m, mu_a, c_m = 0, c_a = 0):
    '''
    Essa função determina o valor inicial de C0 baseado no ponto de equilíbrio livre de 
    doença do modelo.

    :params Ms: int. População de mosquitos suscetíveis. 
    :params k: float. Fraction of female mosquitoes hatched from all egs
    :params delta: float. Average oviposition rate - day^-1
    :params gamma_m: float. Average aquatic transition rate - day^-1
    :params mu_a: float. Average aquatic mortality rate - day^-1
    :params mu_m: float. Average mosquito mortality rate - day^-1
    :params c_m: float. Control effort rate on terretrial phase
    :params c_a: float. Control effort rate on aquatic phase

    :returns: float. 

    '''
    R = R_m(k,delta,gamma_m, mu_m,mu_a, c_m = c_m, c_a = c_a)
    
    C = (R*A0(Ms, gamma_m, mu_m, c_m))/(R - 1)

    return C

def A0(Ms, gamma_m, mu_m, c_m):
    '''
    Essa função determina o valor inicial de mosquitos na fase aquática baseado no ponto de equilíbrio livre de 
    doença do modelo.

    :params Ms: int. População de mosquitos suscetíveis. 
    :params gamma_m: float. Average aquatic transition rate - day^-1
    :params mu_m: float. Average mosquito mortality rate - day^-1
    :params c_m: float. Control effort rate on terretrial phase

    :returns: float. 
    
    '''

    A = (Ms*(mu_m + c_m))/(gamma_m)

    return A

def R_m(k,delta,gamma_m, mu_m, mu_a, c_m = 0, c_a = 0):
    '''
    Essa função the  'basic offspring' da população de mosquitos. Ele é determinado a partir
    do ponto de equilíbrio livre de doença do modelo. 

    :params k: float. Fraction of female mosquitoes hatched from all egs
    :params delta: float. Average oviposition rate - day^-1
    :params gamma_m: float. Average aquatic transition rate - day^-1
    :params mu_a: float. Average aquatic mortality rate - day^-1
    :params mu_m: float. Average mosquito mortality rate - day^-1
    :params c_m: float. Control effort rate on terretrial phase
    :params c_a: float. Control effort rate on aquatic phase

    :returns: float. 
    
    '''

    Rm = (k*delta*gamma_m)/( (mu_m + c_m)*(gamma_m + mu_a + c_a) )

    return Rm

def sup_cap_yang(df, k=7,w1=0.5, C0=5,C1=30,C2 =0.1):
    '''
    Função que realiza o cálculo da capacidade suporte variando de acordo com os dados 
    climáticos seguindo a formulação do Yang. 

    :params df: pd.DataFrame. O dataframe, obrigatoriamente, deve ter as colunas:
                             - "daily_precipitation-mm"
                             - "temp_min-celsius"
                             - "temp_mean-celsius"

    :params k = 7: número de dias anteriores que a chuva vai afetar
    :params w1 = 0.5: efeito residual de chuvas passadas [1/ºC]
    :params C0 = 5: Capacidade da chuva de produzir novos breeding sites
    :params C1 = 30: Quanticade crítica de chuva na formação de breeding sites [mm]
    :params C2 = 0.1: Variação independente nos breeding sites
    '''
    

    #chuva, temperatura mínima, máxima no dia j
    W = df["daily_precipitation-mm"]
    T_min = df["temp_min-celsius"]
    T_max = df["temp_mean-celsius"]
    W_m = []

    n = df.shape[0]

    #números abaixo de 7 é 2015 !!
    for j in range(k,n,1):#passagem dos dias a partir de k dias j=7 é dia 1
        #somatório para o W_m(chuvas dos dias anteriores ao j-th dia)
        soma = 0
        for i in range(1,k+1,1):# somatória sempre de 1 até k dias anteriores
            soma += (W[j-i])/(w1*(T_max[j-i]+T_min[j-i]))**i #summation

        W_m.append(soma) #W_m[0] será j=7

    #Os k primeiros valores do C são NaN
    C = C2 + (C0*(W[k:]+ W_m))/(C1 + W[k:] + W_m)
   
    return C 


def C(t, D, cap_t):
    '''
    Função que retorna um valor para a capacidade suporte.

    :params t: float. Determina o instante de tempo considerado.
    :params D: int. Determina a magnitude da capacidade suporte.
    :params cap_t: float or array. Irá multiplicar com (10**D) para determinar o valor
                   da capacidade suporte no instante t. 

    :returns: float. 
    '''

    if isinstance(cap_t, numbers.Number):
        cap = (10**D)*cap_t
    else:
        cap = (10**D)*cap_t[int(t)]

    return cap


def get_temp(start_date, end_date):
    '''
    Função que retorna um array com a temperatura média em um determinado intervalo de 
    tempo.

    :params start_date: string. Data no formato: %Y-%m-%d.
    :params end_date: string. Data no formato: %Y-%m-%d.

    :returns: array.
    '''
    
    df_we = get_weather_data()
    
    df_we = df_we.loc[(df_we.index >= start_date) & (df_we.index <= end_date)]
    
    return df_we['temp_mean-celsius'].values

def theta_m(t, temp, fixed): 
    '''
    Função que retorna um valor para theta_m baseado na temperatura seguindo os
    valores salvos em um dicionário previamente calculado usando os trabalhos do Yang.

    :params t: float. Determina o instante de tempo considerado.
    :params temp: array. Array com as temperaturas para os respectivos dias.
    :params fixed: boolean. Se True, a função vai retornar o mesmo valor independente de t. 

    :returns: float. 
    '''

    if fixed == True:
        par = 0.11
    else: 
        par = dict_theta_m[temp[int(t)]]

    return par

def gamma_m(t,temp, fixed):
    '''
    Função que retorna um valor para gamma_m baseado na temperatura seguindo os
    valores salvos em um dicionário previamente calculado usando os trabalhos do Yang.

    :params t: float. Determina o instante de tempo considerado.
    :params temp: array. Array com as temperaturas para os respectivos dias.
    :params fixed: boolean. Se True, a função vai retornar o mesmo valor independente de t. 

    :returns: float. 
    ''' 

    if fixed == True:
        par = 0.095
    else:
        par = dict_gamma_m[temp[int(t)]]
    
    return par

def mu_a(t, temp, fixed):
    '''
    Função que retorna um valor para mu_a baseado na temperatura seguindo os
    valores salvos em um dicionário previamente calculado usando os trabalhos do Yang.

    :params t: float. Determina o instante de tempo considerado.
    :params temp: array. Array com as temperaturas para os respectivos dias.
    :params fixed: boolean. Se True, a função vai retornar o mesmo valor independente de t. 

    :returns: float. 
    '''

    if fixed == True:
        par = 0.24
    else:
        par = dict_mu_a[temp[int(t)]]

    return par

def mu_m(t, temp, fixed):
    '''
    Função que retorna um valor para mu_m baseado na temperatura seguindo os
    valores salvos em um dicionário previamente calculado usando os trabalhos do Yang.

    :params t: float. Determina o instante de tempo considerado.
    :params temp: array. Array com as temperaturas para os respectivos dias.
    :params fixed: boolean. Se True, a função vai retornar o mesmo valor independente de t. 

    :returns: float. 
    '''

    if fixed == True:
        par = 0.055

    else:
        par = dict_mu_m[temp[int(t)]]
    
    return par

def d(t,temp, fixed):
    '''
    Função que retorna um valor para delta baseado na temperatura seguindo os
    valores salvos em um dicionário previamente calculado usando os trabalhos do Yang.

    :params t: float. Determina o instante de tempo considerado.
    :params temp: array. Array com as temperaturas para os respectivos dias.
    :params fixed: boolean. Se True, a função vai retornar o mesmo valor independente de t. 

    :returns: float. 
    '''

    if fixed == True:
        par = 5.6
    else:
        par = dict_d[temp[int(t)]]

    return par


def system_odes(t,x, param_fit, param_fixed, temp, cap, fixed = True):
    '''
    Função que implementa o sistema de equações. 
    :params param_fit: tuple. parâmetros que serão fitados. 
    :params param_fixed: tuple. parâmetros que serão fixados. 
    :params temp: array. Array com os valores de temperatura. O tamanho desse array deve ser 
                        condizente com o intervalo de tempo que o modelo será integrado. 
    :params cap: float or array. Parametro que irá determinar a cap suporte do modelo. 
    :params fixed: boolean. Se True serão usados os parâmetros ontomológicos fixos. 
    '''

    #definindo parâmetros que vão ser fitados
    b, beta = param_fit 

    beta_m = beta
    beta_h = beta

    MU_H, THETA_H, ALPHA_H, K, C_A, C_M, D = param_fixed

    #Colocando cada variável em uma posição:
    A  = x[0] #Aquatic mosquito population
    Ms = x[1] #Susceptible mosquitos population
    Me = x[2] #Exposed mosquitos population (infected but not infectious)
    Mi = x[3] #Infectious mosquitos population
    Hs = x[4] #Susceptible human population
    He = x[5] #Exposed human population
    Hi = x[6] #Infectious human population
    Hr = x[7] #Recovered Individuals


    #Definindo condições de contorno para as variáveis(?):
    M = A+Ms+Me+Mi  #População total de Mosquitos
    H = Hs+He+Hi+Hr #População total de humanos
    
    #Definindo cada ODE:
    dA_dt  = K*d(t,temp, fixed)*(1-(A/C(t, D, cap)))*M - (gamma_m(t,temp, fixed) + mu_a(t,temp, fixed) + C_A)*A
    dMs_dt = gamma_m(t,temp, fixed)*A - (b*beta_m*Ms*Hi)/H - (mu_m(t,temp, fixed) + C_M)*Ms
    dMe_dt = (b*beta_m*Ms*Hi)/H - (theta_m(t,temp, fixed ) + mu_m(t,temp, fixed) + C_M)*Me
    dMi_dt = theta_m(t,temp, fixed)*Me - (mu_m(t,temp, fixed) + C_M)*Mi
    dHs_dt = MU_H*(H-Hs) - (b*beta_h*Hs*Mi)/H
    dHe_dt = (b*beta_h*Hs*Mi)/H - (THETA_H + MU_H)*He
    dHi_dt = THETA_H*He - (ALPHA_H + MU_H)*Hi
    dHr_dt = ALPHA_H*Hi - MU_H*Hr

    return [dA_dt, dMs_dt, dMe_dt, dMi_dt, dHs_dt, dHe_dt, dHi_dt, dHr_dt]


def solve_model(t, y0, param_fit, param_fixed, temp, cap, fixed):
    '''
    Função que computa a solução numérica do sistema de equações. 
    
    :params t: array. Intervalo de tempo que deverá ser computado.
    :params y0: list or array. Deve conter os valores das condições iniciais do modelo. 
    :params param_fit: tuple. parâmetros que serão fitados. 
    :params param_fixed: tuple. parâmetros que serão fixados. 
    :params temp: array. Array com os valores de temperatura. O tamanho desse array deve ser 
                        condizente com o intervalo de tempo que o modelo será integrado. 
    :params cap: float or array. Parametro que irá determinar a cap suporte do modelo. 
    :params fixed: boolean. Se True serão usados os parâmetros ontomológicos fixos. 
    '''

    
    r  = solve_ivp(system_odes, t_span = [ t[0], t[-1]], y0 = y0, t_eval = t, args=(param_fit, param_fixed, temp, cap, fixed)) 

    return r 


def solve_fit(out, t, y0, temp, df_we = None, fixed = False): 
    '''
    Retorna a saída do modelo com os parâmetros fitados. 

    :params out: saída do lmfit. 
    :params t: array. Intervalo de tempo que deverá ser computado. 
    :params y0: list or array. Deve conter os valores das condições iniciais do modelo. 
    :params temp: array. Array com os valores de temperatura. O tamanho desse array deve ser 
                        condizente com o intervalo de tempo que o modelo será integrado. 
    :params df_we: pd.Dataframe or None. No caso de um dataframe será computado as capacidade
                    suporte usando a fórmula do Yang. 
    :params fixed: boolean. Se True serão usados os parâmetros ontomológicos fixos. 
    '''
    pars = out.params
    pars = pars.valuesdict()
    b_f = pars['b']
    beta_f = pars['beta']

    parametros_fitting = b_f, beta_f 

    # parâmetros fixos 
    MU_H = 1/(365*67)    #human mortality rate - day^-1
    ALPHA_H = 0.1 #recovering rate - day^-1
    THETA_H = 0.027   #intrinsic incubation rate - day^-1
    K = 0.5          #fraction of female mosquitoes hatched from all egs
    C_A = 0.0      #control effort rate on aquatic phase
    C_M = 0.0    #control effort rate on terretrial phase
    D = 4 

    if isinstance(df_we, pd.DataFrame):

        c_f = sup_cap_yang(df_we)
    else: 
        c_f = pars['c']

    par_fixed = MU_H, THETA_H, ALPHA_H, K, C_A, C_M, D
    
    r_fit  = solve_model(t, y0, parametros_fitting, par_fixed, temp, c_f, fixed ) 

    return r_fit.y[6] + r_fit.y[7]

def plot_fit(t, data, fit):
    '''
    Função para comparar a curva fitada com os dados.
    :params t: array. Intervalo de tempo. 
    :params data: array. dados.
    :params fit: array. A curva fitada. 
    '''
    fig, ax = plt.subplots()
    #plot of fitted function
    ax.plot(t, fit, color='blue',label='Fitted Model')

    #plot of data
    ax.scatter(t,data,color='black',label='Dados')

    #Set the labels
    ax.set_ylabel('Infected cases')
    ax.set_xlabel('Dias')

    #Create a grid for visualization
    ax.grid()

    #Set the title
    ax.set_title('Dengue outbreak ')
    #The size of the numbers on the axixis

    plt.legend()
    plt.show()
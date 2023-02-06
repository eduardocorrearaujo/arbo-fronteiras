import pandas as pd 
from datetime import timedelta
import matplotlib.pyplot as plt 

def get_dengue_data(mean = True):

    '''
    Essa função faz o download dos dados de dengue previamente limpos pelo Alex. 
    São retornadas as séries temporais de casos notificados, prováveis e confirmados
    em laboratório. 
    :params mean: boolean. If True, é aplicada uma média móvel de 7 dias nos dados.
    '''

    data = pd.read_csv('https://raw.githubusercontent.com/AlertaDengue/arbo-fronteiras/main/data/dengue_cases-2010_2022.csv', index_col = 'date')

    data.index = pd.to_datetime(data.index)

    # correções segunda Alex 
    data['notified'] = data['notified'] + data['probable'] + data['lab_confirmed']

    data['probable'] = data['probable'] + data['lab_confirmed']

    if mean:
        data = data.rolling(window = 7).mean().dropna()

        data['acum_notified'] = data.notified.cumsum()

    return data 

def parse_date(date):
    
    """Transforma os dados no formato correto, acrescendo 0 no início de alguns dias e
    meses do ano para garantir que o pd.to_datetime não vai interpretar a data de forma 
    equivocada
    """
    
    new_date = ''
    
    for i in date.split('/'): 

        if len(i) == 1:

            new_date = new_date + '0'+ i + '/' 

        else:

            new_date =  new_date + i + '/' 
        
    new_date = new_date[:-1]
    
    return new_date


def fill_nan_weather(df):
    '''
    Essa função foi criada para corrigir os dados de temperatura. Na ausência de valores nulos ele vai verificar se a temperatura mínima e máxima é dif de zero
    em caso afirmativo, irá substituir esse valor pela média dos últimos 7 dias 

    :params df: pd.Dataframe. O dataframe de entrada, obrigatoriamente, deve ter as seguintes colunas:
                            * temp_min-celsius
                            * temp_max-celsius

    '''


    if df.isnull().sum().sum() == 0:
        for i in df.loc[ (df['temp_min-celsius'] == 0) & (df['temp_max-celsius'] == 0)  ].index:
            # média dos últimos 7 dias
            df.loc[i] = df.loc[i - timedelta(7): i - timedelta(1)].mean()

    else: 
        for col in df.columns: 
            for i in df.loc[df[col].isna() == True].index: 
                df.loc[i][col] = df.loc[i - timedelta(7): i - timedelta(1)][col].mean()
        
        if df.isnull().sum().sum() == 0:
            for i in df.loc[ (df['temp_min-celsius'] == 0) & (df['temp_max-celsius'] == 0)  ].index:
                # média dos últimos 7 dias
                df.loc[i] = df.loc[i - timedelta(7): i - timedelta(1)].mean()

    return df

def get_weather_data():
    ''''
    Essa função carrega os dados climáticos salvos pelo Alex no github.
    '''

    we_data = pd.read_csv('https://raw.githubusercontent.com/AlertaDengue/arbo-fronteiras/main/data/weather-2010_2022.csv')
    
    we_data['date'] = we_data['date'].apply(lambda x: parse_date(x))

    we_data.set_index('date', inplace = True)

    we_data.index = pd.to_datetime(we_data.index)

    for col in ['daily_precipitation-mm', 'temp_max-celsius',         
                'temp_min-celsius', 'temp_mean-celsius',        
                'mean_relative_humidity-%', 'mean_wind_speed-m_per_s']:
    
        we_data[col] = pd.to_numeric(we_data[col],
                                 errors = 'coerce')
    
    we_data = fill_nan_weather(we_data)

    return we_data 

def plot_data(df):
    '''
    Essa função plota os dados de casos notificados diários e notificados lado a lado. 
    :params df: pd.DataFrame. O dataframe obrigatoriamente deve ter as colunas:
                            * `notified`
                            * `acum_notified`
    '''
    fig, ax = plt.subplots(1,2, figsize = (12,5))

    ax[0].plot(df.notified)

    ax[0].set_title('Casos Notificados diários')
    ax[0].grid()

    for tick in ax[0].get_xticklabels():
            tick.set_rotation(45)

    ax[1].plot(df.acum_notified)

    ax[1].set_title('Casos Notificados acumulados')
    ax[1].grid()

    for tick in ax[1].get_xticklabels():
            tick.set_rotation(45)


    plt.show()

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

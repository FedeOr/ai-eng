<<<<<<< HEAD
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
import datetime
import pyodbc
from azure.storage.blob import BlobServiceClient, BlobClient

from PIL import Image

def connect_DB(tableName):
    
    #cnxn = pyodbc.connect('Driver={ODBC Driver 13 for SQL Server};Server=tcp:aiengserver.database.windows.net,1433;Database=CampusData;Uid=ai_user;Pwd=P@ssw0rdR&ti01!;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30')
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=tcp:aiengserver.database.windows.net,1433;'
                      'Database=CampusData;'
                      'Uid=AzureUser;'
                      'Pwd=PasswordReti01')

    cursor = conn.cursor()

    df_database = pd.read_sql_query('SELECT top 5 * FROM dbo.'+ tableName, conn)
    
    return df_database
    
def dataset_transform(df_log):
    # Valore di TelegramType da mantenere
    ttype = 'write      '
    # Creazione dizionario
    d = {'W':1, '%':2, 'ppm':3, 'C°':4, 'Wh':5, 'bool':6}
    # Rimozione righe con TelegramType = 'response'
    df_log = df_log.loc[df_log.TelegramType==ttype]
    # Trasformo i dati boolean in modo da poter essere interpretati dal modello
    df_log.loc[df_log.Value=='False',["Value"]] = 0
    df_log.loc[df_log.Value=='True',["Value"]] = 1
    # Creazione della colonna a partire dal mapping del dizionario
    df_log['IdType'] = df_log['ValueType'].map(d)
    # Conversione colonna 'Data'
    df_log['Data'] = pd.to_datetime(df_log['Data'])
    # Conversione colonna 'Value'
    df_log["Value"] = pd.to_numeric(np.char.replace(df_log["Value"].to_numpy().astype(str),',','.'))
    
    return df_log

def preprocessing(df_log):
    
    # Estraggo l'ora dalla data della rilevazione
    df_log['Hour'] = pd.DatetimeIndex(df_log['Data']).hour
    # Rimozione colonne non necessarie per il training del modello
    column_to_drop = ['Data','IndividualAddress','GroupAddress','TelegramType','ValueType','Description']
    df_log.drop(column_to_drop, axis=1, inplace=True)
    # df_log.drop(column_to_drop)
    
    return df_log

def main_menu():
    try:
        image = Image.open('images/reti_logo.jpg')
        st.image(image, width=150)
        
        st.title("Cosa hanno rilevato i sensori?")
        st.markdown("All'interno del campus sono posizionati numerosi **sensori** per monitorare costantemente"+
                    " le condizioni ambientali all'interno degli edifici che lo costituiscono.  \n"
                    "Con questa applicazione vogliamo"+
                    " mostrare i **dati** che sono stati rilevati e le **anomalie** presenti in essi, ricavate utilizzando un modello"+
                    " di Machine Learning.")
        
        data_expander = st.beta_expander("Dati raccolti")
        with data_expander:
            st.write("Per visualizzare quali sensori sono collocati in ogni edificio utilizza i menu sottostanti:")
            
            df_building = pd.read_csv("data/building.csv")
            df_group = pd.read_csv("data/group.csv", encoding='cp1252')
            
            names = df_building['Nome'].tolist()
            ids = df_building['Id'].tolist()
            dictionary = dict(zip(ids, names))
            building_option = st.selectbox("Scegli l'edificio", ids, format_func=lambda x: dictionary[x])
            
            group_option = st.selectbox("Scegli la grandezza misurata", df_group.ValueType.loc[df_group.IdBuilding == building_option].reset_index(drop=True).unique())

            st.write("Sensori:")
            st.dataframe(df_group.loc[(df_group.IdBuilding == building_option) & (df_group.ValueType == group_option),['GroupAddress', 'Description']])
        
        pipeline_expander = st.beta_expander("Pipeline di predizione")
        with pipeline_expander:
            st.subheader("Pre processing")
            st.write("Per l'analisi degli outlier sono state prese in considerazione solo le rilevazioni mandate automaticamente dai sensori,"+
            " tralasciando quelle inserite manualmente.")
            st.write("È stata aggiunta una colonna *Hour*, estrapolandola dalla colonna Data. La scelta è dovuta al fatto che le"+
                     " grandezze possando dipendere dal momento della giornata in cui sono registrate")
            st.write("È stato inoltre creato un dizionario per decodificare le diverse unità di misura della colonna *ValueType*,"+
                     " poichè il modello scelto prende in input grandezze numeriche.")
            st.subheader("Model")
            st.write("L'algoritmo di Machine Learning che è stato scelto per rilevare le anomalie registrate è **Isolation Forest**."+
                     " Si tratta di un algoritmo di *Anomaly Detection* **non supervisionato**. Per rilevare le anomalie, viene selezionata una grandezza"+
                     " ed eseguito uno split sui dati basandosi su di essa. Procedendo iterativamente in questo processo, verranno segnalati come anomali"+
                     " i dati che saranno isolati con un minor numero di iterazioni.")
            
            link = '[Doc Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest)'
            st.markdown('Per maggiori informazioni: '+link, unsafe_allow_html=True)
            
            image = Image.open('images/IsolationForest1.png')
            st.image(image, width=400)
            
            st.write("Le grandezze che sono state utilizzate per fare il training del modello sono:")
            st.write("ID BUILDING  \n" + "HOUR  \n" + "ID_TYPE  \n" + "VALUE  \n")
            
            st.write("Gli iperparametri utilizzati per addestrare il modello sono stati: *n_estimators* e *max_samples*."+
                     " La combinazione di iperparametri migliore, utilizzata nel modello salvato è *n_estimators=* e *max_samples=*")
        
        prediction_expander = st.beta_expander("Predizioni")
        with prediction_expander:
            st.write("Le predizioni effettuate sul dataset utilizzato per fare il training del modello, hanno evedenziato le seguenti"+
                     " proporzioni tra *Inlier* e *Outlier*")
            
            outliers = 2.657845#df['Prediction'].loc[df.Prediction == -1].count()/df['Prediction'].count()
            labels = 'Outliers', 'Inliers'
            sizes = [30,70]#df.groupby('Prediction').count().Data
            explode = (0.1, 0)

            fig, ax = plt.subplots(figsize=(10,5))
            ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 15})
            ax.axis('equal')
            
            st.pyplot(fig)
            
            st.write("Per effettuare la predizione degli *outlier* sui nuovi dati utilizzare il bottone nella barra laterale.")
    
    except Exception as e:
        print(f'Error: {e}')
        return False
          
def side_menu():
    table_name = st.sidebar.text_input("Inserire il nome della tabella su cui fare la nuova predizione")
    
    if(st.sidebar.button('Esegui Predizione')):
        with st.sidebar:
            try:
                if table_name == "":
                    st.error("Errore: nome tabella non valido")
                else:
                    
                    df = connect_DB(table_name)
                    #df = pd.read_csv("data\log2.csv", sep=';', encoding='cp1252')
                    
                    df_transform = dataset_transform(df)
                    
                    df_preprocess = preprocessing(df_transform)
                                
                    prediction = score_model(df_preprocess)
                    df['Prediction'] = prediction
                    
                    json_result = df.to_json(orient='split')
                    save_json(json_result)
                                
                    outliers = df['Prediction'].loc[df.Prediction == -1].count()/df['Prediction'].count() * 100
                    st.write('La predizione effettuata sui nuovi dati, ha rilevato la presenza di outlier nella misura del %1.1f%%.' % outliers)
                    labels = 'Outliers', 'Inliers'
                    sizes = df.groupby('Prediction').count().Value
                    if sizes.count() == 1:
                        sizes[0] = 0
                    explode = (0.1, 0)

                    fig, ax = plt.subplots(figsize=(15,10))
                    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 25})
                    ax.axis('equal')
                    
                    st.pyplot(fig)
            
            except:
                st.error(f'Errore durante la predizione.')
        
def score_model(data: pd.DataFrame):
    
    model_url = "https://adb-4483624067336826.6.azuredatabricks.net/model/Team2-IsoForest/Production/invocations"#os.environ.get("MODEL_URL")
    headers = {'Authorization': f'Bearer {"dapi183eecc004e8a34a083cc8389d6836b4"}'} #{'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
    data_json = data.to_dict(orient='split')
    response = requests.request(method='POST', headers=headers, url=model_url, json=data_json)

    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

def save_json(json_data: str):
    conn_string ="DefaultEndpointsProtocol=https;AccountName=storageaccountaieng92cc;AccountKey=yRee7zuNsdVCv2AFf/MhrGd8eGfOPncQvDfmXYN3F9/wQ9QCj+9RE0k1r+kXtehudbWDNgZ+3cQqGKFVivgWKg==;EndpointSuffix=core.windows.net"#"AZURE_STORAGE_ACCOUNT"
    container = "team2"#("BLOB_CONTAINER_NAME"
    
    utc_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    json_filename = f"prediction_{utc_timestamp}.json"
    
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    blob_client = blob_service_client.get_blob_client(container=container, blob=json_filename)
    blob_client.upload_blob(json_data)

if __name__ == "__main__":
    main_menu()
    side_menu()
    
=======
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
import datetime
from azure.storage.blob import BlobServiceClient, BlobClient

from PIL import Image

def dataset_transform(df_log):
    # Valore di TelegramType da mantenere
    ttype = 'write      '
    # Creazione dizionario
    d = {'W':1, '%':2, 'ppm':3, 'C°':4, 'Wh':5, 'bool':6}
    # Rimozione righe con TelegramType = 'response'
    df_log = df_log.loc[df_log.TelegramType==ttype]
    # Trasformo i dati boolean in modo da poter essere interpretati dal modello
    df_log.loc[df_log.Value=='False',["Value"]] = 0
    df_log.loc[df_log.Value=='True',["Value"]] = 1
    # Creazione della colonna a partire dal mapping del dizionario
    df_log['IdType'] = df_log['ValueType'].map(d)
    # Conversione colonna 'Data'
    df_log['Data'] = pd.to_datetime(df_log['Data'])
    # Conversione colonna 'Value'
    df_log["Value"] = pd.to_numeric(np.char.replace(df_log["Value"].to_numpy().astype(str),',','.'))
    
    return df_log

def preprocessing(df_log):
    
    # Estraggo l'ora dalla data della rilevazione
    df_log['Hour'] = pd.DatetimeIndex(df_log['Data']).hour
    # Rimozione colonne non necessarie per il training del modello
    column_to_drop = ['Data','IndividualAddress','GroupAddress','TelegramType','ValueType','Description']
    df_log.drop(column_to_drop, axis=1, inplace=True)
    # df_log.drop(column_to_drop)
    
    return df_log

def main_menu():
    try:
        image = Image.open('images/reti_logo.jpg')
        st.image(image, width=150)
        
        st.title("Cosa hanno rilevato i sensori?")
        st.markdown("All'interno del campus sono posizionati numerosi **sensori** per monitorare costantemente"+
                    " le condizioni ambientali all'interno degli edifici che lo costituiscono.  \n"
                    "Con questa applicazione vogliamo"+
                    " mostrare i **dati** che sono stati rilevati e le **anomalie** presenti in essi, ricavate utilizzando un modello"+
                    " di Machine Learning.")
        
        data_expander = st.beta_expander("Dati raccolti")
        with data_expander:
            st.write("Per visualizzare quali sensori sono collocati in ogni edificio utilizza i menu sottostanti:")
            
            df_building = pd.read_csv("data/building.csv")
            df_group = pd.read_csv("data/group.csv", encoding='cp1252')
            
            names = df_building['Nome'].tolist()
            ids = df_building['Id'].tolist()
            dictionary = dict(zip(ids, names))
            building_option = st.selectbox("Scegli l'edificio", ids, format_func=lambda x: dictionary[x])
            
            group_option = st.selectbox("Scegli la grandezza misurata", df_group.ValueType.loc[df_group.IdBuilding == building_option].reset_index(drop=True).unique())

            st.write("Sensori:")
            st.dataframe(df_group.loc[(df_group.IdBuilding == building_option) & (df_group.ValueType == group_option),['GroupAddress', 'Description']])
        
        pipeline_expander = st.beta_expander("Pipeline di predizione")
        with pipeline_expander:
            st.subheader("Pre processing")
            st.write("Per l'analisi degli outlier sono state prese in considerazione solo le rilevazioni mandate automaticamente dai sensori,"+
            " tralasciando quelle inserite manualmente.")
            st.write("È stata aggiunta una colonna *Hour*, estrapolandola dalla colonna Data. La scelta è dovuta al fatto che le"+
                     " grandezze possando dipendere dal momento della giornata in cui sono registrate")
            st.write("È stato inoltre creato un dizionario per decodificare le diverse unità di misura della colonna *ValueType*,"+
                     " poichè il modello scelto prende in input grandezze numeriche.")
            st.subheader("Model")
            st.write("L'algoritmo di Machine Learning che è stato scelto per rilevare le anomalie registrate è **Isolation Forest**."+
                     " Si tratta di un algoritmo di *Anomaly Detection* **non supervisionato**. Per rilevare le anomalie, viene selezionata una grandezza"+
                     " ed eseguito uno split sui dati basandosi su di essa. Procedendo iterativamente in questo processo, verranno segnalati come anomali"+
                     " i dati che saranno isolati con un minor numero di iterazioni.")
            
            link = '[Doc Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest)'
            st.markdown('Per maggiori informazioni: '+link, unsafe_allow_html=True)
            
            image = Image.open('images/IsolationForest1.png')
            st.image(image, width=400)
            
            st.write("Le grandezze che sono state utilizzate per fare il training del modello sono:")
            st.write("ID BUILDING  \n" + "HOUR  \n" + "ID_TYPE  \n" + "VALUE  \n")
            
            st.write("Gli iperparametri utilizzati per addestrare il modello sono stati: *n_estimators* e *max_samples*."+
                     " La combinazione di iperparametri migliore, utilizzata nel modello salvato è *n_estimators=* e *max_samples=*")
        
        prediction_expander = st.beta_expander("Predizioni")
        with prediction_expander:
            st.write("Le predizioni effettuate sul dataset utilizzato per fare il training del modello, hanno evedenziato le seguenti"+
                     " proporzioni tra *Inlier* e *Outlier*")
            
            outliers = 2.657845#df['Prediction'].loc[df.Prediction == -1].count()/df['Prediction'].count()
            labels = 'Outliers', 'Inliers'
            sizes = [30,70]#df.groupby('Prediction').count().Data
            explode = (0.1, 0)

            fig, ax = plt.subplots(figsize=(10,5))
            ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 15})
            ax.axis('equal')
            
            st.pyplot(fig)
            
            st.write("Per effettuare la predizione degli *outlier* sui nuovi dati utilizzare il bottone nella barra laterale.")
    
    except Exception as e:
        print(f'Error: {e}')
        return False
          
def side_menu():
    if(st.sidebar.button('Esegui Predizione')):
        with st.sidebar:
            
            df = pd.read_csv("data\log2.csv", sep=';', encoding='cp1252')
            #st.dataframe(df)
            df_transform = dataset_transform(df)
            
            df_preprocess = preprocessing(df_transform)
            #st.dataframe(df_preprocess)
                        
            prediction = score_model(df_preprocess)
            df['Prediction'] = prediction
            
            json_result = df.to_json(orient='split')
            save_json(json_result)
                        
            outliers = df['Prediction'].loc[df.Prediction == -1].count()/df['Prediction'].count() * 100
            st.write('La predizione effettuata sui nuovi dati, ha rilevato la presenza di outlier nella misura del %1.1f%%.' % outliers)
            labels = 'Outliers', 'Inliers'
            sizes = df.groupby('Prediction').count().Value
            if sizes.count() == 1:
                sizes[0] = 0
            explode = (0.1, 0)

            fig, ax = plt.subplots(figsize=(15,10))
            ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 25})
            ax.axis('equal')
            
            st.pyplot(fig)
        
def score_model(data: pd.DataFrame):
    
    model_url = "https://adb-4483624067336826.6.azuredatabricks.net/model/Team2-IsoForest/Production/invocations"#os.environ.get("MODEL_URL")
    headers = {'Authorization': f'Bearer {"dapi183eecc004e8a34a083cc8389d6836b4"}'} #{'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
    data_json = data.to_dict(orient='split')
    response = requests.request(method='POST', headers=headers, url=model_url, json=data_json)

    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

def save_json(json_data: str):
    conn_string ="DefaultEndpointsProtocol=https;AccountName=storageaccountaieng92cc;AccountKey=yRee7zuNsdVCv2AFf/MhrGd8eGfOPncQvDfmXYN3F9/wQ9QCj+9RE0k1r+kXtehudbWDNgZ+3cQqGKFVivgWKg==;EndpointSuffix=core.windows.net"#"AZURE_STORAGE_ACCOUNT"
    container = "team2"#("BLOB_CONTAINER_NAME"
    
    utc_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    json_filename = f"prediction_{utc_timestamp}.json"
    
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    blob_client = blob_service_client.get_blob_client(container=container, blob=json_filename)
    blob_client.upload_blob(json_data)

if __name__ == "__main__":
    main_menu()
    side_menu()
    
>>>>>>> 26db1409eb43718633a5cdd01134f0f43805babe

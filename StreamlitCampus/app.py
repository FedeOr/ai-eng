import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from PIL import Image
from src.features import read_data, preprocessing
from src.model import predict_model
from src.prediction import save_prediction
from load_settings import get_setting

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
            
            df_building = pd.DataFrame(read_data.read_DB_table("Building"))
            df_group = pd.DataFrame(read_data.read_DB_table("Group"))
        
            names = df_building['Nome'].tolist()
            ids = df_building['Id'].tolist()
            dictionary = dict(zip(ids, names))
            building_option = st.selectbox("Scegli l'edificio", ids, format_func=lambda x: dictionary[x])
            
            group_option = st.selectbox("Scegli la grandezza misurata", df_group.ValueType.loc[df_group.IdBuilding == building_option].reset_index(drop=True).unique())

            st.write("Sensori:")
            st.dataframe(df_group.loc[(df_group.IdBuilding == building_option) & (df_group.ValueType == group_option),['GroupAddress', 'Description']])
        
        pipeline_expander = st.beta_expander("Training del modello")
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

    except Exception as e:
        print(f'Error: {e}')
        return False
          
def side_menu():
    table_name = st.sidebar.text_input("Inserire il nome della tabella su cui fare la predizione")
    
    if(st.sidebar.button('Esegui Predizione')):
        with st.sidebar:
            try:
                if table_name == "":
                    st.error("Errore: nome tabella non valido")
                else:
                    
                    df = read_data.read_DB_table(table_name)
                    
                    df_transform = preprocessing.dataset_transform(df)
                    df_preprocess = preprocessing.preprocessing(df_transform)
                    prediction = predict_model.score_model(df_preprocess)
                    df['Prediction'] = prediction
                    
                    json_result = df.to_json(orient='split')
                    utc_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    json_filename = f"prediction_{utc_timestamp}.json"
                    save_prediction.save_json(json_result, json_filename)
                    
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
                    
                    st.markdown(f'Per maggiori dettagli, è stato salvato nello storage il file *{json_filename}*')
            
            except Exception as e:
                st.error(f'Errore durante la predizione.{e}')

if __name__ == "__main__":
    main_menu()
    side_menu()
import pandas as pd
import pyodbc

from load_settings import get_setting

def read_DB_table(tableName):
    
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server='+get_setting("SERVER")+';'
                      'Database='+get_setting("DATABASE")+';'
                      'Uid='+get_setting("DB_USER")+';'
                      'Pwd='+get_setting("DB_PWD")+';'
                      'Connection Timeout=30')

    cursor = conn.cursor()

    df_database = pd.read_sql_query('SELECT * FROM [dbo].['+ tableName+']', conn)
    
    return df_database
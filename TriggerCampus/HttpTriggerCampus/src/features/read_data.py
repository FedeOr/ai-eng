import pandas as pd
import pyodbc
import os

def read_DB_table(tableName):
    
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server='+os.environ.get("SERVER")+';'
                      'Database='+os.environ.get("DATABASE")+';'
                      'Uid='+os.environ.get("DB_USER")+';'
                      'Pwd='+os.environ.get("DB_PWD")+';'
                      'Connection Timeout=30')

    cursor = conn.cursor()

    df_database = pd.read_sql_query('SELECT TOP 5 * FROM [dbo].['+ tableName+']', conn)
    
    return df_database
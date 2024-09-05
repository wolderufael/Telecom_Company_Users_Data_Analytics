import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2

# Load environment variables from .env file
load_dotenv()

# Fetch credentials from .env file
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

class Connector:
# Connect to your PostgreSQL database
    def load_table_to_dataframe(self,table_name):
        try:
            connection = psycopg2.connect(
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
                database=db_name
            )

            # Use pandas read_sql_query to load the table into a DataFrame
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql_query(query, connection)

            return df

        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)

        finally:
            # Close the connection
            if connection:
                connection.close()
                print("PostgreSQL connection is closed")
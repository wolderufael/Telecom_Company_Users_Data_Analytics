# This is used for task 1 and 2
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

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
    def load_table_to_dataframe(self, table_name):
        try:
            # Create an SQLAlchemy engine
            engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

            # Use pandas read_sql to load the table into a DataFrame
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, engine)

            return df

        except Exception as error:
            print("Error while connecting to PostgreSQL", error)

        finally:
            # Close the connection
            engine.dispose()
            print("SQLAlchemy connection is disposed")
# for task1,2 and 3
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
    def add_dataframe_to_table(self, df, table_name, if_exists='replace'):
        try:
            # Create SQLAlchemy engine for database connection
            engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

            # Use pandas to_sql to add DataFrame to the database
            df.to_sql(table_name, engine, index=False, if_exists=if_exists)

            print(f"DataFrame successfully added to table '{table_name}' in the database.")

        except Exception as error:
            print("Error while inserting DataFrame to PostgreSQL", error)

        finally:
            if engine:
                engine.dispose()
                print("SQLAlchemy engine is disposed.")
                
    def load_table_to_dataframe(self, table_name):
        try:
            self.connection_string=f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
            #Create SQLAlchemy engine for database connection
            engine = create_engine(self.connection_string)
            
            # Load the data into a pandas DataFrame
            dataframe = pd.read_sql_table(table_name, engine)
            return dataframe
        
        except Exception as error:
            print("Error while connecting to PostgreSQL", error)
        
        finally:
            # Only dispose the engine if it was successfully created
            if 'engine' in locals():
                engine.dispose()
                print("SQLAlchemy connection is disposed")

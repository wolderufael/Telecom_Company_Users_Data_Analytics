<<<<<<< HEAD
# This is used for task 1 and 2
=======
# for task1,2 and 3
>>>>>>> Task-3
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
<<<<<<< HEAD
=======

>>>>>>> Task-2
# Load environment variables from .env file
load_dotenv()

# Fetch credentials from .env file
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

class Connector:
<<<<<<< HEAD

    def add_dataframe_to_table(self, df, table_name, if_exists='fail'):
=======
    def add_dataframe_to_table(self, df, table_name, if_exists='replace'):
>>>>>>> Task-4
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
# Connect to your PostgreSQL database
    def load_table_to_dataframe(self, table_name):
        try:
            # Create an SQLAlchemy engine
            engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

>>>>>>> Task-2
            # Use pandas read_sql to load the table into a DataFrame
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, engine)

            return df

        except Exception as error:
            print("Error while connecting to PostgreSQL", error)

        finally:
            # Close the connection
            engine.dispose()
<<<<<<< HEAD
            print("SQLAlchemy connection is disposed")
                
        # Add DataFrame to PostgreSQL table
=======
>>>>>>> Task-3
    def add_dataframe_to_table(self, df, table_name, if_exists='fail'):
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
<<<<<<< HEAD
=======
            print("SQLAlchemy connection is disposed")
>>>>>>> Task-2
=======
                
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
>>>>>>> Task-3

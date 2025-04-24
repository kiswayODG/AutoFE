import pandas as pd
from io import StringIO
import streamlit as st

def load_uploaded_csv_files(file):
  
        try:
           
            try:
                content = file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                content = file.getvalue().decode("latin1")

            df = pd.read_csv(StringIO(content))
            return df
        except Exception as e:
            st.error(f"Erreur de lecture du fichier {file.name} : {e}")



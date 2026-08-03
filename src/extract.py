import pandas as pd
path="data/Exposure_SOV_practice.xlsx"

df=pd.read_excel(path,sheet_name="Broker B - Castlegate")
text = df.to_string(index=False,na_rep="")
print(text)
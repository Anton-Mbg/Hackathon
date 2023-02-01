import os
import pandas as pd
import numpy as np

path_input_data = os.path.join('.', 'test_combined_Class.csv')
# If we trained our model using the Order taxonomic level. If not, it can be for example test_combined_Species.csv
df_data = pd.read_csv(path_input_data, delimiter=',')
df_data.drop(df_data.columns[0], axis=1, inplace=True)
# We drop the index column

##########################
# You do the preprocessing here
##########################
def stdNormalize(df):
    for col in df.columns[0:-1]:
        colMean = df[col].mean()
        colSTD = np.std(df[col])
        for i in range(len(df[col])):
            df[col][i] = (df[col][i] - colMean) / colSTD
    return df


df_data = df_data[['Bacteria;Bacteroidota;Bacteroidia',
                   'Bacteria;Bdellovibrionota;Bdellovibrionia',
                   'Bacteria;Campilobacterota;Campylobacteria',
                   'Bacteria;Dependentiae;Babeliae',
                   'Bacteria;Elusimicrobiota;Elusimicrobia',
                   'Bacteria;Fibrobacterota;Fibrobacteria',
                   'Bacteria;Firmicutes;uncultured',
                   'Bacteria;Fusobacteriota;Fusobacteriia',
                   'Bacteria;Myxococcota;Polyangia', 'Bacteria;Myxococcota;bacteriap25',
                   'Bacteria;Patescibacteria;Saccharimonadia',
                   'Bacteria;Spirochaetota;Brachyspirae',
                   'Bacteria;Acidobacteriota;Subgroup 19',
                   'Bacteria;Acidobacteriota;Vicinamibacteria',
                   'Bacteria;Chloroflexi;Chloroflexia',
                   'Bacteria;SAR324 clade(Marine group B);uncultured delta proteobacterium',
                   'label']]

df_data = stdNormalize(df_data)

path_output_data = path_input_data
df_data.to_csv(path_output_data)

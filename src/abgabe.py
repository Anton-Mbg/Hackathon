import os
import pandas as pd

path_input_data = os.path.join('.', 'train_combined_Class.csv')
# If we trained our model using the Order taxonomic level. If not, it can be for example test_combined_Species.csv
df_data = pd.read_csv(path_input_data, delimiter=',')
df_data.drop(df_data.columns[0], axis=1, inplace=True)
# We drop the index column

##########################
# You do the preprocessing here
##########################

path_output_data = path_input_data
df_data.to_csv(path_output_data)

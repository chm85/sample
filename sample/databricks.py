# Databricks notebook source
# MAGIC %md
# MAGIC #### Purpose: 
# MAGIC This module applies historic demand distributions to the PA forecast in order to map the forecast to geographic regions. This dataset is then used for customers that are allowed to **split ship**.  
# MAGIC   
# MAGIC **Example:** K87-BLK split is 45% **Non Split** and 55% **Split**. Then the DC team wants to replicate in the following DCs (RCV - Primary, DC3 and DC5 as the alternate).  
# MAGIC   
# MAGIC **Definitions**  
# MAGIC - Primary DC: This DC only ships the **Non Split** demand for a given replicated material.
# MAGIC - Alternate DC: This DC only ships the **Split** demand for a given replicated material.  
# MAGIC   
# MAGIC The DC Team has set DC3 to the following 15 states (ME, NH, VT, MA, RI, CT, NY, NJ, PA, DE, MD, VA, NC, SC, GA),therefore, DC5 would pick up the remaining 35 states. Then we would sum up the totals from the state columns in the dataset below and then adjust the score to sum to the **55%** of the split demand.
# MAGIC - RCV - Primary: This DC would only hold the 45% for the Non Split  
# MAGIC **Reproportion the percentages**
# MAGIC - DC3 - Alternate: 30%*55% = 16.5%
# MAGIC - DC5 - Alternate: 70%*55% = 38.5%
# MAGIC - Total Percentage = 55% + 16.5% + 38.5% = 100% of the K87-BLK Demand
# MAGIC
# MAGIC ##### Note: This dataset only shows a subset of states but that actual dataframe has all 50 states.
# MAGIC
# MAGIC ##### Definitions
# MAGIC - states [ak - wy]:
# MAGIC   - Description: Represents the percentage distribution of materials shipped to each of the 50 U.S. states.
# MAGIC   - Details:
# MAGIC     - For every material, this module provides the calculated percentage that was shipped to each state.
# MAGIC     - The cumulative sum of the percentages for all 50 states will always equal 100%.
# MAGIC - grouping: There are new materials that we do not have history on or materials that we have history but we did not sell many units leaving it to be unreliable. Therefore, we use history to apply geographic demand to these new materials.
# MAGIC   - Example: for material K87-BLK we would use its history but if had a new color created for K87 then we would use the stylecode average to obtain our demand percentages by state. 
# MAGIC - pa_forecast: The total units forecasted for a given material. This is not subset to those who can and cannot split ship.
# MAGIC   - Date Range: please see config.yaml
# MAGIC - total_units: The total units shipped for customers that are allowed to split ship. 
# MAGIC   - Date Range: please see config.yaml
# MAGIC
# MAGIC |segment  |split         |category   |subcategory|stylecodeid|material  |ak    |al    |ar    |az    |wa    |wi    |wv    |wy    |total_units|grouping|pa_forecast|
# MAGIC |---------|--------------|-----------|-----------|-----------|----------|------|------|------|------|------|------|------|------|-----------|--------|-----------|
# MAGIC |Wholesale|Split Customer|accessories|headwear   |101195     |101195-001|0.0039|0.0084|0.0028|0.0151|0.0384|0.0395|0.0912|0.0017|49930      |history |153750     |
# MAGIC |Wholesale|Split Customer|tops       |knit shirts|K126       |K126-HGY  |0.0016|0.0153|0.0039|0.0274|0.0182|0.0153|0.0338|0.0052|138061     |history |130294     |
# MAGIC |D2C      |Split Customer|tops       |knit shirts|K231       |K231-PRT  |0.0015|0.0114|0.0062|0.0184|0.0132|0.0209|0.0059|0.0026|2722       |history |8028       |
# MAGIC |D2C      |Split Customer|accessories|headwear   |100289     |100289-001|0.002 |0.0167|0.0076|0.0169|0.0199|0.0126|0.0041|0.0003|3423       |history |10633      |
# MAGIC |Wholesale|Split Customer|tops       |knit shirts|104616     |104616-G73|0.002 |0.0154|0.0043|0.06  |0.014 |0.0325|0.0446|0.0036|184926     |history |83225      |
# MAGIC
# MAGIC

# COMMAND ----------

import mlflow
import yaml
import numpy as np
import pandas as pd
from pyspark.shell import spark
import json
from inventorium.utils import dbx_schema
from inventorium.constants import CONSTANTS, US_STATES
from inventorium.janitor import jan_clean_columns, jan_remove_whitespace, jan_value_casing, jan_rename_cols
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# Open the YAML file and load it into a Python dictionary
with open('../../inventorium/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set database
DATABASE = CONSTANTS['global']['Database']

# Expierment name
EXPERIMENT_NAME = CONSTANTS['global']['mlflow']

def historical_averages(dataset : pd.core.frame.DataFrame,columns_to_average, grp_by : list) ->  pd.core.frame.DataFrame:

    agg_dict = {}
    for state in columns_to_average:
        agg_dict[state] = 'mean'

    dataset = dataset.groupby(grp_by).agg(agg_dict).reset_index().copy()	
    return dataset
        
# Save Datasets
SOURCE_SCRIPT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[-1]
FOLDER_NAME = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[-2]

if __name__ == "__main__":

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=SOURCE_SCRIPT):

        spark.sql(DATABASE)

        # Read and process data
        pa_dataset = spark.read.table("1_etl_step_1_process_demand_data").toPandas().fillna(0)
        state_dataset = spark.read.table("1_etl_step_1_process_state").toPandas().fillna(0)
        state_dataset = state_dataset.query("split == 'Split Customer'").copy()
        ri_items =  spark.read.table("2_project_setup_step_0_1_replicated_items").toPandas()

        # Save plan name
        plan_name = str(pa_dataset['savedplanname'].unique()[0])
        pa_plan = {'pa_plan':plan_name}
        with open('/dbfs/FileStore/inventory_replication/replicated_inv_user_form_json/pa_plan.json', 'w', encoding='utf-8') as f:
          json.dump(pa_plan, f, ensure_ascii=False, indent=4)

        # if we do not have at least a 500 or 2500 (by stylecode) units of demand then history will be off and it will be better to use the subcat demand history
        lower_case_list = [x.lower() for x in US_STATES]
        removals = state_dataset[['stylecodeid','total']].groupby('stylecodeid').sum().reset_index().query("total < 2500") 
        removals = list(removals.stylecodeid)
        state_dataset = state_dataset.query(f"material not in {removals}").copy()
        state_dataset = state_dataset.query("total > 500").copy()
        state_dataset[lower_case_list] = state_dataset[lower_case_list].apply(lambda x: round(x / x.sum(),3), axis=1)
        state_dataset = state_dataset.sort_values("total", ascending=False).reset_index(drop = True)

        # lightly process columns
        state_dataset = (state_dataset.copy()
            .jan_clean_columns()
            .jan_remove_whitespace()  
            .jan_value_casing(['category','subcategory'],case='lower')
            .jan_value_casing(['material'],case='upper')
            .jan_rename_cols({"distributionchannel":"segment"})
        )
        state_dataset['segment'] = state_dataset['segment'].replace("CCG Industrial","CCG")
        
        # we do not need segment yet so we will roll everything up to the material
        pa_dataset = pa_dataset.groupby(['segment','category','subcategory','stylecodeid','material']).sum().reset_index()

# COMMAND ----------

# MAGIC %md ##### Join Replicated items and PA Dataset as we only want to process replicated items
# MAGIC

# COMMAND ----------

ri_materials = list(ri_items.material.unique())
pa_dataset = pa_dataset.query(f"material in {ri_materials}").copy()

# COMMAND ----------

# MAGIC %md ##### Check what new materials we need to process
# MAGIC

# COMMAND ----------

# subset to join to historic_dataset so we know what materials to remove
check_pa = pa_dataset[['segment','material']].drop_duplicates().copy()
check_pa['exists'] = 1
check_dataset = state_dataset[['segment','material']].drop_duplicates().copy()
check_dataset['exists'] = 1

state_dataset = pd.merge(state_dataset,check_pa,how = 'left')
state_dataset = state_dataset.fillna('Does not exist in PA')

pa_dataset = pd.merge(pa_dataset,check_dataset,how = 'left')
pa_dataset = pa_dataset.fillna('Add to historic_dataset')

# Remove records that do not exist in PA
pa_categories = list(pa_dataset['category'].unique())
missing_categorie_dataset = state_dataset.query(f"category not in {pa_categories}")
state_dataset = state_dataset.query("exists == 1 ")
state_dataset = pd.concat([state_dataset,missing_categorie_dataset])
retired_products_dataset = state_dataset.query("exists != 1").copy()

# add pa records to historic_dataset
pa_new_dataset = pd.DataFrame()
pa_subset = pa_dataset.query("exists == 'Add to historic_dataset'").copy()
pa_subset['row_id'] = pa_subset['stylecodeid'] + pa_subset['segment']

# COMMAND ----------

# MAGIC %md #####Identify new materials where we need to create geographic demand based upon averages
# MAGIC

# COMMAND ----------

historical_style_averages = historical_averages(dataset = state_dataset,columns_to_average = lower_case_list,grp_by = ['stylecodeid','segment'])
historical_subcat_averages = historical_averages(dataset = state_dataset,columns_to_average = lower_case_list,grp_by = ['subcategory','segment'])
historical_cat_averages = historical_averages(dataset = state_dataset,columns_to_average = lower_case_list,grp_by = ['category','segment'])
# Incase we have records that do match on cat and segment
historical_cat_only_averages = historical_averages(dataset = state_dataset,columns_to_average = lower_case_list,grp_by = ['category'])

# COMMAND ----------

# MAGIC %md ##### Carhartt consistently introduces new materials each year, and for these new additions, there might not be historical data regarding the demand or shipping destination. To address this, we use a tiered approach to predict the demand based on available data:
# MAGIC
# MAGIC - Stylecode: Initially, we try to match and infer demand based on the specific style code of the product. This provides a granular and precise understanding if there's any existing data for that particular style.
# MAGIC
# MAGIC - Subcategory (Subcat): If there's no match or data at the style code level, we then move to the subcategory level. The assumption here is that items within the same subcategory might have similar demand patterns.
# MAGIC
# MAGIC - Category: As a last resort, if we can't find matches at the style code or subcategory levels, we use the broader category data. This gives a more generalized prediction but ensures we have some basis for our demand forecasting for the new materials.

# COMMAND ----------

stylecode_matches = pd.merge(pa_subset,
                             historical_style_averages, 
                             how = 'inner', on = ['stylecodeid','segment'])
stylecode_matches['grouping'] = 'stylecode'
ids = list(stylecode_matches.row_id)

subcat_matches = pd.merge(pa_subset.query(f"row_id not in {ids}"),
                          historical_subcat_averages,
                           how = 'inner', on = ['subcategory','segment'])
subcat_matches['grouping'] = 'subcategory'
ids = ids + list(subcat_matches.row_id)

cat_matches = pd.merge(pa_subset.query(f"row_id not in {ids}"),
                          historical_cat_averages,
                           how = 'inner', on = ['category','segment'])
cat_matches['grouping'] = 'category'
ids = ids + list(cat_matches.row_id)

cat_only_matches = pd.merge(pa_subset.query(f"row_id not in {ids}"),
                          historical_cat_only_averages,
                           how = 'inner', on = ['category'])
cat_only_matches['grouping'] = 'category no segment'
new_items = pd.concat([stylecode_matches,subcat_matches,cat_matches,cat_only_matches])

# COMMAND ----------

# MAGIC %md ##### Change dataframes so we can merge
# MAGIC

# COMMAND ----------

# bind in new material to historic historic_dataset
new_items['split'] = 'Split Customer'
new_items = (new_items.copy()
            .jan_rename_cols({'pa_forecast':'total_units'})
        )
state_dataset = (state_dataset.copy()
            .jan_rename_cols({'total':'total_units','exists':'grouping'})
        )
new_items = new_items[state_dataset.columns]
state_dataset['grouping'] = np.where(state_dataset['grouping'] ==1, 'history',state_dataset['grouping'])

# COMMAND ----------

state_dataset = pd.concat([state_dataset,new_items])

# COMMAND ----------

# MAGIC %md #####Remove materials that are not in the current demand plan
# MAGIC

# COMMAND ----------

removal = pd.merge(state_dataset[['material']],pa_dataset[['material','subcategory']],how = 'left').fillna(0).query("subcategory == 0")
removal_li = list(removal.material.unique())
state_dataset = state_dataset.query(f"material not in {removal_li}")
# Merge to get the pa forecast
state_dataset = pd.merge(state_dataset,pa_dataset[['segment','material','pa_forecast']].groupby(['segment','material']).sum().reset_index(), on = ['segment','material'])

# COMMAND ----------

# filename	
FOLDER = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[-2]
FILENAME_SEGMENT = FOLDER +"_"+ SOURCE_SCRIPT  	

# Extract Schema	
state_dataset , schema = dbx_schema(dataset=state_dataset )	

# Save Data	
state_dataset  = spark.createDataFrame(state_dataset , schema)	
state_dataset.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FILENAME_SEGMENT)

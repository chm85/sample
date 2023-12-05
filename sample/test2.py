# Databricks notebook source
# MAGIC %md #####Purpose: This module is designed to select materials that would be good candidates for replication based upon a ruleset created by Courtney S. Chris S. and Monica V.
# MAGIC

# COMMAND ----------

import pandas as pd
import mlflow
from inventorium.conn import conn_str
from inventorium.constants import CONSTANTS
from inventorium.utils import dbx_schema
from shutil import move
from datetime import datetime

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
pd.set_option('display.max_columns', None)

# summary page
START = 0
START_LARGE = 30000
WHS_VALUE = 50000
CCG_D2C_VALUE = 1
WOMENS_VALUE = 50000
KEY = 999999

# Getting the current date
current_date = datetime.now().strftime('%Y_%m_%d')

# Creating the new file name with the current date
file_name = f'standard_{WHS_VALUE}_{current_date}.xlsx'

# Source Connections
JDBC_CONNECTION = ''
DB_CONN = conn_str()
exec(DB_CONN)

# Set database
DATABASE = CONSTANTS['global']['Database']

# Expierment name
EXPERIMENT_NAME = CONSTANTS['global']['mlflow']

# Save Datasets
SOURCE_SCRIPT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[-1]
FOLDER_NAME = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[-2]

def compute_coef_of_variation(data):
    # Group by material and compute mean, standard deviation, and count
    grouped = data.groupby(['distributionchannel','material']).agg({'total': ['mean', 'std', 'count']})
    
    # Compute CV for each material
    grouped[('total', 'coef_of_variation')] = (grouped[('total', 'std')] / grouped[('total', 'mean')]) * 100
    
    # Rename columns for clarity
    grouped.columns = ['mean', 'std_dev', 'count', 'coef_of_variation']
    grouped.reset_index(inplace=True)
    
    return grouped[['distributionchannel','material', 'count', 'coef_of_variation']]

if __name__ == "__main__":

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=SOURCE_SCRIPT):
        
        # Set database
        spark.sql(DATABASE)

        pa_dataset = spark.read.table("1_etl_step_1_process_demand_data").toPandas().fillna(0)
        historic_materials = spark.read.table("1_etl_step_1_process_historic_materials").toPandas().fillna(0)
        replicated_materials = spark.read.table("1_etl_step_1_process_replicated_materials").toPandas().fillna(0)
        drops_dataset = pd.read_excel("/dbfs/FileStore/inventory_replication/drop_list_v2.xlsx")
        demand_variant_data = spark.read.table("1_etl_step_1_process_demand_variant_data").toPandas()
        products = spark.read.table("1_etl_step_0_etl_products").toPandas()
        dataset = spark.read.table('5_primary_pipeline_step_2_safety_stock_dataset').toPandas()
        original_dataset = dataset.copy()

        products['department'] = products['department'].str.replace("'", "")
        
        # correct PFAS 
        pfas_dataset = pd.read_excel("/dbfs/FileStore/inventory_replication/pfas.xlsx")
        pfas_dataset['material_new'] = pfas_dataset['material_new'].str.replace("--","-")
        pfas_dataset_stylecodeid = {
            "stylecodeid": [106672, 106673, 106687, 106652, 106652, 106652, 106652, 106652, 106673, 106673, 106672, 106673, 106665, 106652, 106672, 106652, 106652],
            "material_new": ["106672-BRN", "106673-BRN", "106687-N04", "106652-CRH", "106652-HGY", "106652-N04", "106652-I26", "106652-446", "106673-BLK", "106673-GVL", "106672-DNY", "106673-DNY", "106665-001", "106652-W03", "106672-BLK", "106652-G73", "106652-WHT"]
        }
        
        # Convert to pandas dataframe
        pfas_dataset_stylecodeid = pd.DataFrame(pfas_dataset_stylecodeid)
        pfas_dataset_stylecodeid['stylecodeid'] = pfas_dataset_stylecodeid['stylecodeid'].astype(str)
        pfas_dataset = pd.merge(pfas_dataset,pfas_dataset_stylecodeid)
        pfas_dataset_copy = pfas_dataset.copy()

        # clean drops dataset
        drops_dataset = drops_dataset.fillna(0)
        drops_dataset = drops_dataset.query("material != 0")[['material','fiscal_year','season']]
        drops_dataset['drop'] = True

# COMMAND ----------

# MAGIC %md Remove internally manufactured materials and bibs

# COMMAND ----------

data_list = [
    "106679-BRN", "B136-DKB", "102776-001", "102776-211", "102776-DKB",
    "103828-BLK", "103828-BRN", "104050-BLK", "104050-BRN", "104050-DKB",
    "104050-MOS", "104394-BLK", "106679-BLK", "B136-BLK", "B151-DKH",
    "B151-NVY", "106671-BRN", "106672-BLK", "106672-BRN", "106682-BLK",
    "104392-BLK", "104392-BRN", "106673-BLK", "106673-BRN", "106673-GVL"
]
bibs = ['104672-DST','102987-039']

data_list = data_list + bibs
dataset = dataset.query(f"material not in {data_list}")

# COMMAND ----------

# MAGIC %md Update datasets with PFAS replacements
# MAGIC

# COMMAND ----------

for index, row in pa_dataset.iterrows():
    material = row['material']
    if material in list(pfas_dataset['material']):
        material_new = pfas_dataset[pfas_dataset['material'] == material]['material_new'].iloc[0]
        stylecode_new = pfas_dataset[pfas_dataset['material'] == material]['stylecodeid'].iloc[0]
        pa_dataset.loc[index,'material'] = material_new
        pa_dataset.loc[index,'stylecodeid'] = stylecode_new

for index, row in dataset.iterrows():
    material = row['material']
    if material in list(pfas_dataset['material']):
        material_new = pfas_dataset[pfas_dataset['material'] == material]['material_new'].iloc[0]
        stylecode_new = pfas_dataset[pfas_dataset['material'] == material]['stylecodeid'].iloc[0]
        dataset.loc[index,'material'] = material_new
        dataset.loc[index,'stylecodeid'] = stylecode_new
        # we set this to 0 because  we will use the new items percents
        dataset.loc[index,'parent_percent'] = 0
        dataset.loc[index,'alternate_percent'] = 0

# COMMAND ----------

dataset = dataset[['segment', 'category', 'subcategory', 'stylecodeid', 'material',
       'adjustment', 'parent_percent', 'alternate_percent', 'pa_forecast']]
# sum to rollup the pfas items to a single record
dataset = dataset.groupby(['segment', 'category', 'subcategory', 'stylecodeid', 'material']).agg({
    'parent_percent': 'sum',
    'alternate_percent': 'sum',
    'pa_forecast': 'sum'
}).reset_index()

# COMMAND ----------

# MAGIC %md #####How many materials?
# MAGIC

# COMMAND ----------

print("There are {} unique wholesale materials in the demand plan".format(len(dataset.material.unique())))

# COMMAND ----------

# MAGIC %md #####Count Variants
# MAGIC

# COMMAND ----------

demand_variant_data = demand_variant_data[['material','materialvariant']].drop_duplicates().groupby('material').count().reset_index().rename(columns={'materialvariant': 'variant_count'})

# COMMAND ----------

# MAGIC %md #####Subset to records where we have favorable split between non-split and split customers
# MAGIC   * We will select a range between 25% & 75%
# MAGIC

# COMMAND ----------

pivot_df = dataset[['segment','category','subcategory','stylecodeid','material','parent_percent','alternate_percent','pa_forecast']]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Check for duplicates

# COMMAND ----------

pivot_df['id'] = pivot_df['segment'] + pivot_df['material']
pivot_df['ct'] = 1
duplicate_ct = len(pivot_df[['id','ct']].groupby('id').sum().reset_index().query("ct > 1"))

# COMMAND ----------

# MAGIC %md #####Check History
# MAGIC   * How long have we been selling this product and how consistent is the demand
# MAGIC     * The lookback period is 4 years
# MAGIC   * Note: Carhartt does not have Like-For-Like products so as an example PFAS replacements or slight color variations will register as a new material. 
# MAGIC     * One example would be the force shirt with the new enhancements it is an entirely new material
# MAGIC   * coef_of_variation = std_dev / mean_sales
# MAGIC   * CV > 1: If the CV is greater than 1, it means that the relative variability is high compared to the mean of the data. This often indicates that the data is relatively inconsistent or spread out.
# MAGIC

# COMMAND ----------

# MAGIC %md #####Replace historic materials with the PFAS replacements
# MAGIC

# COMMAND ----------

historic_materials_update = pd.merge(historic_materials,pfas_dataset,how = 'inner', on = 'material')
historic_update_materials = list(historic_materials_update.material.unique())
historic_materials = historic_materials.query(f"material not in {historic_update_materials}").copy()

# update old material with new and drop column so we can append
historic_materials_update['material'] = historic_materials_update['material_new']
del historic_materials_update['material_new']
del historic_materials_update['pfas']
historic_materials = pd.concat([historic_materials,historic_materials_update])

pfas_dataset['material'] = pfas_dataset['material_new']
del pfas_dataset['material_new']

# COMMAND ----------

result = compute_coef_of_variation(historic_materials).fillna(9999999).rename(columns={'count': 'n_years_sold','distributionchannel': 'segment'}).reset_index()
del result['index']
pivot_df = pd.merge(pivot_df, result, on = ['segment','material'],how = 'left')
pivot_df['coef_of_variation'] = round(pivot_df['coef_of_variation'],2)

# COMMAND ----------

# MAGIC %md #####Create a volume feature
# MAGIC   * This looks at the total demand it buckets into sizes small, medium, large, and xl
# MAGIC     * small:0-4999
# MAGIC     * medium:5000-19,999
# MAGIC     * large:20,00 -74,999
# MAGIC     * xl: >= 75,000
# MAGIC

# COMMAND ----------

small_stop = 5000
medium_stop = START_LARGE
large_stop = 75000
xl_start = 75001

# Define the value ranges and corresponding categories
value_ranges = [float('-inf'), small_stop, medium_stop, large_stop, xl_start, float('inf')]
# not a typo Bin labels must be one fewer than the number of bin edges
categories = ['small', 'medium', 'large', 'xl', 'xl' ]

# Create the 'bin_category' column using pd.cut() with ordered=False
pivot_df['bin_category'] = pd.cut(pivot_df['pa_forecast'], bins=value_ranges, labels=categories, right=False, ordered=False)

# COMMAND ----------

# MAGIC %md #####Verify Seasons
# MAGIC

# COMMAND ----------

season = pa_dataset[['segment','material','season']].groupby(['segment','material']).count().reset_index().rename(columns={'season': 'season_ct'})
pivot_df = pd.merge(pivot_df,season, on = ['segment','material'],how = 'left')

# COMMAND ----------

# MAGIC %md #####Merge Drops, Variants, Department, and PFAS

# COMMAND ----------

pivot_df = pd.merge(pivot_df,pfas_dataset, how = 'left', left_on = 'material',right_on = 'material').fillna(0)
pivot_df = pd.merge(pivot_df,drops_dataset, how = 'left', left_on = 'material',right_on = 'material').fillna(0)
pivot_df = pd.merge(pivot_df,demand_variant_data, how = 'left', left_on = 'material',right_on = 'material').fillna(0)
pivot_df = pd.merge(pivot_df,products[['department','material']], how = 'left', left_on = 'material',right_on = 'material').fillna(0)
pivot_df = pd.merge(pivot_df,replicated_materials, how = 'left', left_on = 'material',right_on = 'material').fillna(0)

# COMMAND ----------

# MAGIC %md #####Recommendations
# MAGIC

# COMMAND ----------

#TODO: Add note on fiscal year this actually means it is being dropped in 2024

# COMMAND ----------

pivot_df['fiscal_year'] = pivot_df['fiscal_year'].astype(int)
output = pivot_df.query(" category != 'bottoms' and department not in ('Personal Protective') and fiscal_year != 2024")

# COMMAND ----------

# MAGIC %md #####Remove where parent is > .75 for the wholesale segment

# COMMAND ----------

materials = list(output.query("parent_percent > .75 and segment == 'Wholesale'").material)
# this provides visability into the materials that are excluded due to parent demand being > .75 or having no demand
output_copy = output.copy()

output = output.query(f"material not in {materials}")

# COMMAND ----------

whs = output.query(f"pa_forecast > {WHS_VALUE} and segment == 'Wholesale' and department != 'Womens'")
ccg_d2c = output.query(f"pa_forecast > {CCG_D2C_VALUE} and segment != 'Wholesale' and department != 'Womens'")
womens = output.query(f"pa_forecast > {WOMENS_VALUE} and department == 'Womens'  ")

output = pd.concat([whs,ccg_d2c,womens])
output = output.query("parent_percent <= .75 ")

# COMMAND ----------

# MAGIC %md ####Wholesale controls who can and cannot split therefore now we subset this dataset to the selected wholesale materials

# COMMAND ----------

start_count = len(list(output['material'].unique()))
mats = list(output.query("segment == 'Wholesale'")['material'].unique())
output = output.query(f"material in  {mats}").copy()
end_count = len(list(output['material'].unique()))
print("Start {} end {}".format(start_count,end_count))

# COMMAND ----------

# MAGIC %md ##### Bring in missing materials we should have the identical materials for all segments even if we do not have demand for those materials. 
# MAGIC

# COMMAND ----------

output_new = output.copy()
output_new['status'] = 'Replication Criteria Met'
tmp = output.query("segment == 'Wholesale'")[['segment','material']].drop_duplicates()
for row in tmp.itertuples():
    material = row.material
    for segment in ['CCG','D2C']:
        # if it is aleady in the output dataset we move on
        tmp_data = output.query(f"segment == '{segment}' and material == '{material}'")
        if len(tmp_data) == 1:
            continue
        else:
            # if it is not in output dataset we check the copy prior to us filtering where parent_percent > .75
            tmp_data = output_copy.query(f"segment == '{segment}' and material == '{material}'").copy()
            tmp_data['parent_percent'] = 1
            tmp_data['alternate_percent'] = 0
            tmp_data['status'] = 'For visibility only'
            if len(tmp_data) == 1:
                output_new = pd.concat([output_new,tmp_data])
            else:
                # These are records where we have no demand in the demand plan
                tmp_data = output_copy.query(f"segment == 'Wholesale' and material == '{material}'").copy()
                tmp_data['segment'] = segment
                tmp_data[['parent_percent','alternate_percent','pa_forecast','season_ct','variant_count','coef_of_variation','n_years_sold','fiscal_year','season']] = 0
                tmp_data['alternate_percent'] = 0
                tmp_data['status'] = 'For visibility only - No demand'
                output_new = pd.concat([output_new,tmp_data])

# COMMAND ----------

agg = output_new[['segment','ct']].groupby('segment').sum().reset_index()
agg['check'] = agg['ct'] == end_count

if agg.check.sum() != 3:
    raise Exception("segment values are not matching")
output = output_new.copy()

# COMMAND ----------

# MAGIC %md ##### Aggregate segments
# MAGIC

# COMMAND ----------

pivot_data = output[['segment','pa_forecast']].groupby('segment').sum().reset_index()
pivot_data = {pivot_data['segment'][i]: pivot_data['pa_forecast'][i] for i in range(len(pivot_data['segment']))}
# Create a new DataFrame from the pivoted data
pivot_data = pd.DataFrame([pivot_data])

# COMMAND ----------

# MAGIC %md #####Summary

# COMMAND ----------


query_filter = "category != 'bottoms' and department != 'Personal Protective' and fiscal_year != 2024"
d2c_mats = len(output.query("segment == 'D2C'").query("status == 'Replication Criteria Met' ").material.unique())
ccg_mats = len(output.query("segment == 'CCG'").query("status == 'Replication Criteria Met' ").material.unique())
whs_mats = len(output.query("segment == 'Wholesale'").query("status == 'Replication Criteria Met' ").material.unique())

mats = list(output.material.unique())
replicated_material_ct = len(replicated_materials.query(f"material in {mats}"))

single_season_items = len(output.query("season_ct == 1").query("status == 'Replication Criteria Met' "))
single_season_items_demand = output.query("season_ct == 1").query("status == 'Replication Criteria Met' ").pa_forecast.sum()
if START == 0:
    start_percentage = '0%'
    end_percentage = '100%'
else:
    start_percentage = "{:.4}%".format(START * 100)
    end_percentage = "{:.4}%".format(END * 100)
run = start_percentage + "-" + end_percentage
total_materials = len(list(output['material'].unique()))
subset_total = output.query("status == 'Replication Criteria Met' ").pa_forecast.sum()
total_demand = pa_dataset.pa_forecast.sum()
demand_percent = round(subset_total/total_demand,3)
tmp = pd.DataFrame({'filter ID': KEY,
'run':'Hybrid', 
'number of replicated materials':replicated_material_ct,
'number of single season materials': single_season_items,
'number of D2C materials': d2c_mats,
'number of CCG materials': ccg_mats,
'number of WHS materials': whs_mats,
'single season materials demand': single_season_items_demand,
'large starting range start': START_LARGE,
'large starting range end': 74999,
'total materials': total_materials,
'total demand': subset_total, 
'pa total demand': total_demand,
'percent of demand plan':demand_percent,
'applied filter':query_filter},[0])

result = pd.concat([tmp, pivot_data], axis=1)

# add missing columns
if 'CCG' not in result.columns:
    result['CCG'] = 0  
if 'D2C' not in result.columns:
    result['D2C'] = 0
    
result = result[['filter ID', 'run','number of replicated materials', 'number of single season materials','number of D2C materials','number of CCG materials','number of WHS materials',
       'single season materials demand', 'large starting range start',
       'large starting range end', 'total materials',
       'percent of demand plan', 'CCG', 'D2C', 'Wholesale','total demand','pa total demand' ,'applied filter']]
result.columns = [x.replace(" ","_") for x in result.columns]

# COMMAND ----------

# convert to string so we can write to Hive
output = output.astype(str)
result = result.astype(str)
output['alternate_percent'] = output['alternate_percent'].astype(float)
output['parent_percent'] = output['parent_percent'].astype(float)
output['pa_forecast'] = output['pa_forecast'].astype(float).astype(int)
result['total_demand'] = result['total_demand'].astype(float).astype(int)

# COMMAND ----------

# MAGIC %md ##### Excluded materials
# MAGIC

# COMMAND ----------

excluded_materials = original_dataset[['segment','category','subcategory','stylecodeid','material','parent_percent','alternate_percent','6400_3PL_Dallas','pa_forecast']].sort_values("pa_forecast",ascending = False)
excluded_materials = pd.merge(excluded_materials, products[['department','material']], how = 'inner', left_on = 'material',right_on = 'material').fillna(0)
df1 =  output[['segment','material']].copy()
# Merge the two dataframes on 'segment' and 'material' columns
merged_df = df1.merge(excluded_materials, on=['segment', 'material'], how='left', indicator=True)

# Keep only the rows from df1 that are not in df2 based on the indicator column
filtered_df1 = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')

# Now, filtered_df1 contains only the rows from df1 that are not in df2 based on 'segment' and 'material'

# If you want to remove records from df2 that are in df1, you can do the following:
filtered_df2 = excluded_materials.merge(df1, on=['segment', 'material'], how='left', indicator=True)
filtered_df2 = filtered_df2[filtered_df2['_merge'] == 'left_only'].drop(columns='_merge')
filtered_df2 = pd.merge(filtered_df2,pfas_dataset_copy, on = 'material', how = 'left').fillna(0).query("material_new == 0")
filtered_df2 = filtered_df2.query(f"material not in {data_list}")
filtered_df2 = filtered_df2.drop(columns=['material_new', 'stylecodeid_y', 'pfas']).rename(columns={'stylecodeid_x': 'stylecodeid'})
filtered_df2 = filtered_df2.query("parent_percent < .75 and category != 'bottoms' and subcategory != 'bib overalls' ").sort_values("pa_forecast",ascending = False).head(100)

# COMMAND ----------

# MAGIC %md ##### Join back to the original dataset

# COMMAND ----------

del original_dataset['pa_forecast']
output.rename(columns={'stylecodeid_x': 'stylecodeid'}, inplace=True)
output = pd.merge(output[['pa_forecast','category','subcategory','stylecodeid','segment','material','status']], original_dataset,on = ['category','subcategory','stylecodeid','segment','material'], how = 'left').fillna(0)

# COMMAND ----------

drop_cols = ['total','row_sum','diff','adjusted_diff','do_not_replicate_y','adjusted_row_sum','adjustment','do_not_replicate_x']
for col in drop_cols:
    del output[col]
output['parent'] = output['parent'].replace(0,'No Primary DC')

# COMMAND ----------

# move to top
DCS = ['1000_RCV_Distribution_Center','6000_3PL_Memphis','6100_3PL_Columbus','6300_Carhartt_DHL_DC_OH3','6400_3PL_Dallas']
PREFIX_DC = ['adjusted_' + item for item in DCS]

for row in output.query("status == 'For visibility only' ").itertuples():
    parent = row.parent
    prefix_parent = 'adjusted_' + parent
    for dc in DCS+PREFIX_DC:
        output.at[row.Index, dc]  = 0
    output.at[row.Index, parent]  = 1
    output.at[row.Index, prefix_parent]  = 1

# COMMAND ----------

datasets = {'summary':result,'recommendations':output.query("status == 'Used in PA' ")}

for key in datasets.keys():

    tmp = datasets[key]

    tmp_df, schema = dbx_schema(tmp)

    # Save Data
    tmp_df = spark.createDataFrame(tmp_df, schema)
    
    # filename
    filename = FOLDER_NAME + "_" + SOURCE_SCRIPT + "_" + key

    tmp_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(filename)

# COMMAND ----------

# MAGIC %md ###### Report for the Planning and Distribution team to review selected materials
# MAGIC

# COMMAND ----------

# join in department
output = pd.merge(output,products[['department','material']], how = 'left', left_on = 'material',right_on = 'material').fillna(0)

# Move Department from fron to back
# Step 1: Identify the name of the last column
last_column = output.columns[-1]

# Step 2: Store the data of the column and drop it from the DataFrame
column_data = output[last_column]
output = output.drop(columns=last_column)

# Step 3: Insert the column at the desired position (here, position 3)
# Note: Positions are zero-indexed, so position 3 is actually the fourth place
output.insert(1, last_column, column_data)

# COMMAND ----------

output = output[['status','department', 'category', 'subcategory', 'stylecodeid',
       'segment', 'material','pa_forecast',   '1000_RCV_Distribution_Center',
       '6000_3PL_Memphis', '6100_3PL_Columbus', '6300_Carhartt_DHL_DC_OH3',
       '6400_3PL_Dallas', 'parent', 'parent_percent', 'alternate_percent',
       'adjusted_1000_RCV_Distribution_Center', 'adjusted_6000_3PL_Memphis',
       'adjusted_6100_3PL_Columbus', 'adjusted_6300_Carhartt_DHL_DC_OH3',
       'adjusted_6400_3PL_Dallas', 'safety_units', 'safety_stock_adjustment']]

# COMMAND ----------

test = output.copy()
cols_to_update = ['1000_RCV_Distribution_Center', '6000_3PL_Memphis', '6100_3PL_Columbus','6300_Carhartt_DHL_DC_OH3','6400_3PL_Dallas']
test[cols_to_update] = test[cols_to_update].multiply(test['pa_forecast'], axis='index')

df_long = pd.melt(test, id_vars='segment', value_vars=['1000_RCV_Distribution_Center', '6000_3PL_Memphis', '6100_3PL_Columbus','6300_Carhartt_DHL_DC_OH3','6400_3PL_Dallas'], 
                   value_name='test')
df_long['test'] = df_long['test']

print(df_long.test.sum())
df_long = df_long.groupby(['segment','variable']).sum().reset_index().sort_values("variable")
df_long['test'] = round(df_long['test'] ,3)

# COMMAND ----------

agg = output_new[['status','ct']].groupby('status').sum().reset_index()
agg.rename(columns={'ct': 'value'}, inplace=True)
agg.rename(columns={'status': 'code'}, inplace=True)

logging = pd.DataFrame(({'WHS_VALUE':WHS_VALUE,'CCG_D2C_VALUE':CCG_D2C_VALUE,'WOMENS_VALUE':WOMENS_VALUE,'Duplicates':duplicate_ct}),[0])
logging = pd.melt(logging, var_name='code', value_name='value')

logging = pd.concat([logging, agg])

df_long.rename(columns={'test': 'ct'}, inplace=True)
df_long.rename(columns={'variable': 'distribution_center'}, inplace=True)

# COMMAND ----------

with pd.ExcelWriter(f"/databricks/driver/{file_name}", engine="xlsxwriter") as writer:
    result.to_excel(writer, sheet_name='Result', index=False)
    output.to_excel(writer, sheet_name='All Detail', index=False)
    df_long.to_excel(writer, sheet_name='Agg', index=False)
    filtered_df2.to_excel(writer, sheet_name='Top 100 Excluded Materials', index=False)
    logging.to_excel(writer, sheet_name='Internal Logging', index=False)

move(f"/databricks/driver/{file_name}",  f"/dbfs/temporary/{file_name}")
dbutils.fs.cp( f"dbfs:/temporary/{file_name}",  f"dbfs:/mnt/replicated-inventory-web-interface/exploratory_data_analysis/recommendations/{file_name}")

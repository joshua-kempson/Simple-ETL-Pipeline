import pandas as pd
import numpy as np
import kagglehub

def test(df):
    item_price = {
        "Juice": 3.0,
        "Cake": 3.0,
        "Sandwich": 4.0,
        "Smoothie": 4.0
    }    
    for key, value in item_price.items():
        df.loc[df['Item'] == key, 'Price Per Unit'] = value
    
    if df['Quantity'].isna().any():
        mode_val = df['Quantity'].mode(dropna=True)
        if not mode_val.empty:
            df['Quantity'] = df['Quantity'].fillna(mode_val[0])

    return(df)

def cal_nums(df):
    for index, row in df.iterrows(): 
        quantity = row['Quantity']
        price_per_unit = row['Price Per Unit']
        total_spent = row['Total Spent']

        if pd.isna(quantity) and pd.notna(price_per_unit) and pd.notna(total_spent): 
            df.at[index, 'Quantity'] = total_spent / price_per_unit
        elif pd.isna(price_per_unit) and pd.notna(quantity) and pd.notna(total_spent): 
            df.at[index, 'Price Per Unit'] = total_spent / quantity
        elif pd.isna(total_spent) and pd.notna(quantity) and pd.notna(price_per_unit): 
            df.at[index, 'Total Spent'] = quantity * price_per_unit
    
    return(df)

def num_cols_fill(df):
    item_price = {
        "Coffee": 2.0,
        "Cookie": 1.0,
        "Salad": 5.0,
        "Tea": 1.5
    }

    num_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors = "coerce")
    
    df["Transaction Date"] = df["Transaction Date"].fillna(method="ffill").fillna(method="bfill")

    df = cal_nums(df)

    #not a numbers col cleaning - need to seperate
    for key, value in item_price.items():
        df.loc[df['Price Per Unit'] == value, 'Item'] = key
        df.loc[df['Item'] == key, 'Price Per Unit'] = value

    return(df)

def random_fill_by_group(df, group_col, target_col):
    def fill_group(group):
        prob = group[target_col].value_counts(normalize=True, dropna=True)
        missing = group[target_col].isna()
        if not prob.empty and missing.any():
            group.loc[missing, target_col] = np.random.choice(
                prob.index, size=missing.sum(), p=prob.values
            )
        return group
    
    return df.groupby(group_col).apply(fill_group).reset_index(drop=True)

path = kagglehub.dataset_download("ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training")
org_data = pd.read_csv('/home/joshua/.cache/kagglehub/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training/versions/1/dirty_cafe_sales.csv')

clean_data = org_data.drop_duplicates()

clean_data = clean_data.replace(["ERROR", "UNKNOWN"], pd.NA)

clean_data = num_cols_fill(clean_data)

clean_data = random_fill_by_group(clean_data, "Item", "Payment Method")

clean_data = random_fill_by_group(clean_data, "Item", "Location")

clean_data = test(clean_data)

clean_data = cal_nums(clean_data)

o = org_data.isna().sum()
c = clean_data.isna().sum()

print(f"Original Item NA :{o}")
print(f"Clean Item NA :{c}")
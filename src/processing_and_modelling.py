import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from tqdm import tqdm
import glob
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, roc_curve, confusion_matrix
import pickle
from sklearn.utils import resample

import concurrent.futures
from tqdm import tqdm
from stratify import stratified_subsets
import warnings
import os
import json

import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import zipfile

import matplotlib.pyplot as plt

def ignore_sklearn_deprecation_warning(message, category, filename, lineno, file=None, line=None):
    return category is FutureWarning and "is_sparse is deprecated" in str(message)

warnings.filterwarnings("ignore", category=FutureWarning)






figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)



print('Extracting ECG-View data...')

zip_file_path = "ECG_ViEW_II_for_CVS.zip"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(".")




print('Preprocessing ECG-View data...')
   
df_person = pd.read_csv('ECG_ViEW_II_for_CVS/ECG_ViEW_II_for_CVS/Person.csv')
df_ecg = pd.read_csv('ECG_ViEW_II_for_CVS/ECG_ViEW_II_for_CVS/Electrocardiogram.csv')
df_diagnosis_code = pd.read_csv('ECG_ViEW_II_for_CVS/ECG_ViEW_II_for_CVS/DiagnosisCodeMaster.csv', encoding='unicode_escape')
df_diagnosis = pd.read_csv('ECG_ViEW_II_for_CVS/ECG_ViEW_II_for_CVS/Diagnosis.csv', encoding='unicode_escape')
df_ecg['ecgdate'] = pd.to_datetime(df_ecg['ecgdate'])
df_diagnosis['diagdate'] = pd.to_datetime(df_diagnosis['diagdate'])

df_ecg = df_ecg.merge(df_person, on='personid')
df_diag = df_diagnosis.merge(df_diagnosis_code, on='diaglocalcode')
df_diag = df_diag[df_diag['diagnosis']!='no_data']

df_diag['diagcode'] = df_diag['diagcode'].apply(lambda x: x.replace('.',''))
df_diag = df_diag[['personid','diagdate','diagcode','diagnosis']]

digits = 5
df_diag["diagcode"]=df_diag["diagcode"].apply(lambda x: x[:digits])
df_diag["diagcode"]=df_diag["diagcode"].apply(lambda x: x.rstrip("X"))

def remove_last_letter(code):
    if code[-1].isalpha():
        return code[:-1]
    else:
        return code

df_diag['diagcode'] = df_diag['diagcode'].apply(remove_last_letter)

counts = df_diag['diagcode'].value_counts()
values_to_keep = counts[counts >= 200].index
df_diag = df_diag[df_diag['diagcode'].isin(values_to_keep)]

df_ecg.drop('ACCI',axis=1, inplace=True)
df_ecg.drop('ethnicity',axis=1, inplace=True)
df_ecg.drop('ecgdept',axis=1, inplace=True)
df_ecg.drop('ecgsource',axis=1, inplace=True)
df_diag.drop('diagnosis',axis=1,inplace=True)

birth_year_ranges = {0: (1890, 1894),1: (1895, 1899),2: (1900, 1904),
                     3: (1905, 1909),4: (1910, 1914),5: (1915, 1919),
                     6: (1920, 1924),7: (1925, 1929),8: (1930, 1934),
                     9: (1935, 1939),10: (1940, 1944),11: (1945, 1949),
                     12: (1950, 1954),13: (1955, 1959),14: (1960, 1964),
                     15: (1965, 1969),16: (1970, 1974),17: (1975, 1979),
                     18: (1980, 1984),19: (1985, 1989),20: (1990, 1994),
                     21: (1995, 1999),22: (2000, 2004),23: (2005, 2009),
                     24: (2010, 2013)}

def calculate_age(row):
    # this might give us an error of 1 or 2 years, but it aproximates real number to match external validation
    ecg_year = row['ecgdate'].year
    birth_year_group = row['Birthyeargroup']
    birth_year_range = birth_year_ranges.get(birth_year_group)
    age = ecg_year - np.mean(birth_year_range)
    return age

df_ecg['age'] = df_ecg.apply(calculate_age, axis=1)
df_ecg = df_ecg[df_ecg['age']>=18]


diagnosis_list = []

for i, row in tqdm(df_ecg.iterrows()):
    personid = row['personid']
    ecgdate = row['ecgdate']
    
    df_diags_personid = df_diag[df_diag['personid'] == personid]
    df_diags_personid = df_diags_personid[
        (df_diags_personid['diagdate'] <= ecgdate + pd.Timedelta(days=90)) & 
        (df_diags_personid['diagdate'] >= ecgdate - pd.Timedelta(days=90))
    ]
    
    if not df_diags_personid.empty:
        diagnosis_list.append(df_diags_personid['diagcode'].values.tolist())
    else:
        diagnosis_list.append([])
        
        
df_ecg['diags'] = diagnosis_list

df_view = df_ecg[df_ecg['diags'].map(len) > 0].reset_index(drop=True) # name change

def label_propagation(df, column_name, propagate_all=True):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    col_flat = flatten(np.array(df[column_name]))
    
    def prepare_consistency_mapping_internal(codes_unique, codes_unique_all):
        res={}
        for c in codes_unique:
            if(propagate_all):
                res[c]=[c[:i] for i in range(3,len(c)+1)]
            else: 
                res[c]=np.intersect1d([c[:i] for i in range(3,len(c)+1)],codes_unique_all)
        return res
    
    cons_map = prepare_consistency_mapping_internal(np.unique(col_flat), np.unique(col_flat))
    df[column_name] = df[column_name].apply(lambda x: list(set(flatten([cons_map[y] for y in x]))))
    return df


df_view = label_propagation(df_view, "diags", propagate_all=True)

def process_multilabel(df, column_name, threshold, output_column_name):
    counts = {}
    
    for row in df[column_name]:
        for item in row:
            counts[item] = counts.get(item, 0) + 1
    
    filtered_counts = {item: count for item, count in counts.items() if count >= threshold}
    
    unique_strings = sorted(filtered_counts.keys(), key=lambda x: filtered_counts[x], reverse=True)
    
    df[column_name] = df[column_name].apply(lambda row: [item for item in row if item in filtered_counts])
    
    df[output_column_name] = df[column_name].apply(lambda row: [1 if item in row else 0 for item in unique_strings])

    return df, np.array(unique_strings)



df_view, lbls_view = process_multilabel(df_view, 'diags', 200, 'Diagnoses_labels')

df_view = df_view[df_view['diags'].map(len) > 0].reset_index(drop=True)

age_bins = pd.qcut(df_view['age'], q=4)
unique_intervals = age_bins.cat.categories
bin_labels = {interval: f'{interval.left}-{interval.right}' for interval in unique_intervals}
df_view['age_bin'] = age_bins.map(bin_labels)

df_view['sex'] = df_view['sex'].astype(str)
df_view['merged_strat'] = df_view.apply(lambda row: row['diags'] + [row['age_bin'], row['sex']], axis=1)

col_label = "merged_strat"
col_group = "personid"

res = stratified_subsets(df_view,
               col_label,
               [0.05]*20,
               col_group=col_group,
               label_multi_hot=False,
               random_seed=42)

df_view['strat_fold'] = res
df_view.to_pickle('df_view.pkl')
np.save('lbls_view.npy', lbls_view)


df_view = pd.read_pickle('df_view.pkl')
lbls_view = np.load('lbls_view.npy')
df_view.drop(columns=['Birthyeargroup','age_bin','merged_strat'], inplace=True)

df_view.loc[(df_view['RR'] < 0) | (df_view['RR'] > 5000), 'RR'] = np.nan
df_view.loc[(df_view['PR'] < 0) | (df_view['PR'] > 5000), 'PR'] = np.nan
df_view.loc[(df_view['QRS'] < 0) | (df_view['QRS'] > 5000), 'QRS'] = np.nan
df_view.loc[(df_view['QT'] < 0) | (df_view['QT'] > 5000), 'QT'] = np.nan
df_view.loc[(df_view['QTc'] < 0) | (df_view['QTc'] > 5000), 'QTc'] = np.nan 

df_view.loc[(df_view['P_wave_axis'] < -360) | (df_view['P_wave_axis'] > 360), 'P_wave_axis'] = np.nan
df_view.loc[(df_view['QRS_axis'] < -360) | (df_view['QRS_axis'] > 360), 'QRS_axis'] = np.nan
df_view.loc[(df_view['T_wave_axis'] < -360) | (df_view['T_wave_axis'] > 360), 'T_wave_axis'] = np.nan

df_view['sex'] = df_view['sex'].astype(int)









print('Loading and preprocessing MIMIC data...')


df_mimic = pd.read_csv('records_w_diag_icd10.csv')
df_mimic['all_diag_all'] = df_mimic['all_diag_all'].apply(lambda x:eval(x))
df_mimic['ecg_time'] = pd.to_datetime(df_mimic['ecg_time'])

df_mimic = df_mimic[df_mimic['all_diag_all'].apply(lambda x: len(x) > 0)]

df_mimic["all_diag_all"] = df_mimic["all_diag_all"].apply(lambda x: list(set([y.strip()[:5] for y in x])))
df_mimic["all_diag_all"] = df_mimic["all_diag_all"].apply(lambda x: list(set([y.rstrip("X") for y in x])))
df_mimic['all_diag_all'] = df_mimic['all_diag_all'].apply(lambda code: code[:-1] if code[-1].isalpha() else code)

def label_propagation(df, column_name, propagate_all=True):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    col_flat = flatten(np.array(df[column_name]))
    
    def prepare_consistency_mapping_internal(codes_unique, codes_unique_all):
        res={}
        for c in codes_unique:
            if(propagate_all):
                res[c]=[c[:i] for i in range(3,len(c)+1)]
            else: 
                res[c]=np.intersect1d([c[:i] for i in range(3,len(c)+1)],codes_unique_all)
        return res
    
    cons_map = prepare_consistency_mapping_internal(np.unique(col_flat), np.unique(col_flat))
    df[column_name] = df[column_name].apply(lambda x: list(set(flatten([cons_map[y] for y in x]))))
    return df

def process_multilabel(df, column_name, threshold, output_column_name):
    counts = {}
    
    for row in df[column_name]:
        for item in row:
            counts[item] = counts.get(item, 0) + 1
    
    filtered_counts = {item: count for item, count in counts.items() if count >= threshold}
    
    unique_strings = sorted(filtered_counts.keys(), key=lambda x: filtered_counts[x], reverse=True)
    
    df[column_name] = df[column_name].apply(lambda row: [item for item in row if item in filtered_counts])
    
    df[output_column_name] = df[column_name].apply(lambda row: [1 if item in row else 0 for item in unique_strings])

    return df, np.array(unique_strings)



df_mimic = label_propagation(df_mimic, "all_diag_all", propagate_all=True)
df_mimic, lbls_mimic = process_multilabel(df_mimic, 'all_diag_all', 200, 'Diagnoses_labels')
df_mimic = df_mimic[df_mimic['all_diag_all'].apply(lambda x: len(x) > 0)]


def process_multilabel_mincount(df, column_name, threshold, output_column_name):
    # Step 1: Count occurrences of labels in strat_fold == 18
    counts_fold_18 = {}
    for row in df[df['strat_fold'] == 18][column_name]:
        for item in row:
            counts_fold_18[item] = counts_fold_18.get(item, 0) + 1
    
    # Step 2: Count occurrences of labels in strat_fold == 19
    counts_fold_19 = {}
    for row in df[df['strat_fold'] == 19][column_name]:
        for item in row:
            counts_fold_19[item] = counts_fold_19.get(item, 0) + 1
    
    # Step 3: Filter labels based on threshold in both folds
    filtered_labels = set()
    for label in counts_fold_18:
        if counts_fold_18[label] >= threshold and counts_fold_19.get(label, 0) >= threshold:
            filtered_labels.add(label)
    
    filtered_labels = sorted(filtered_labels, key=lambda x: (counts_fold_18.get(x, 0) + counts_fold_19.get(x, 0)), reverse=True)
    
    # Step 4: Update the DataFrame to keep only the filtered labels
    df[column_name] = df[column_name].apply(lambda row: [item for item in row if item in filtered_labels])
    
    # Step 5: Create the output column based on filtered labels
    df[output_column_name] = df[column_name].apply(lambda row: [1 if item in row else 0 for item in filtered_labels])
    
    return df, np.array(filtered_labels)


df_mimic, lbls_mimic = process_multilabel_mincount(df_mimic, 'all_diag_all', 6, 'Diagnoses_labels')
df_mimic = df_mimic[df_mimic['all_diag_all'].apply(lambda x: len(x) > 0)]
df_mimic.drop(columns=['Unnamed: 0','ed_diag_ed','ed_diag_hosp','hosp_diag_hosp','all_diag_hosp','all_diag_all',
                       'label_strat_all2all','label_test','merged_strat'], inplace=True)
df_mimic['data'] = df_mimic.index



df_machine = pd.read_csv('machine_measurements.csv')
df_machine = df_machine[['study_id','rr_interval', 'p_onset', 'p_end','qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']]




df_mimic = pd.merge(df_mimic, df_machine, on='study_id')

# degrees
df_mimic.loc[(df_mimic['qrs_axis'] < -360) | (df_mimic['qrs_axis'] > 360), 'qrs_axis'] = np.nan
df_mimic.loc[(df_mimic['t_axis'] < -360) | (df_mimic['t_axis'] > 360), 't_axis'] = np.nan
df_mimic.loc[(df_mimic['p_axis'] < -360) | (df_mimic['p_axis'] > 360), 'p_axis'] = np.nan

# msec
df_mimic.loc[(df_mimic['p_onset'] < 0) | (df_mimic['p_onset'] > 5000), 'p_onset'] = np.nan
df_mimic.loc[(df_mimic['p_end'] < 0) | (df_mimic['p_end'] > 5000), 'p_end'] = np.nan
df_mimic.loc[(df_mimic['qrs_onset'] < 0) | (df_mimic['qrs_onset'] > 5000), 'qrs_onset'] = np.nan
df_mimic.loc[(df_mimic['qrs_end'] < 0) | (df_mimic['qrs_end'] > 5000), 'qrs_end'] = np.nan
df_mimic.loc[(df_mimic['t_end'] < 0) | (df_mimic['t_end'] > 5000), 't_end'] = np.nan
df_mimic.loc[(df_mimic['rr_interval'] < 0) | (df_mimic['rr_interval'] > 5000), 'rr_interval'] = np.nan

df_mimic = df_mimic.rename(columns={'rr_interval':'RR',
                                   'p_axis':'P_wave_axis',
                                   'qrs_axis':'QRS_axis',
                                   't_axis':'T_wave_axis',
                                   'gender':'sex'})

df_mimic['sex'] = df_mimic['sex'].apply(lambda x: 1 if x == 'M' else (0 if x == 'F' or x == 0 else np.nan))
df_mimic['PR'] = df_mimic['qrs_onset'] - df_mimic['p_onset']
df_mimic['QRS'] = df_mimic['qrs_end'] - df_mimic['qrs_onset'] 
df_mimic['QT'] = df_mimic['t_end'] - df_mimic['qrs_onset']
df_mimic['QTc'] = np.where(df_mimic['RR'] != 0, df_mimic['QT'] / np.sqrt(df_mimic['RR'] / 1000), np.nan)


df_view.to_pickle('df_view.pkl')
df_mimic.to_pickle('df_mimic.pkl')
np.save('lbls_view.npy', lbls_view)
np.save('lbls_mimic.npy', lbls_mimic)





print('Modelling begings...')


df_view = pd.read_pickle('df_view.pkl')
df_mimic = pd.read_pickle('df_mimic.pkl')
lbls_view = np.load('lbls_view.npy')
lbls_mimic = np.load('lbls_mimic.npy')


def get_plots(x_train, x_val, x_test,
              y_train_m, y_val_m, y_test_m, 
              x_test_ext, y_test_ext_m,
              lbls_internal, lbls_external,
              features, 
              code,
              description, 
              keep_gender=None):
    
    # Find the index of the code in both internal and external labels
    idx_internal = np.where(lbls_internal == code)[0][0]
    idx_external = np.where(lbls_external == code)[0][0]
    
    if keep_gender is not None:
        # Create boolean mask for gender filtering
        train_mask = x_train['Gender'] == keep_gender
        val_mask = x_val['Gender'] == keep_gender
        test_mask = x_test['Gender'] == keep_gender
        test_ext_mask = x_test_ext['Gender'] == keep_gender

        # Filter x_* DataFrames by gender and drop 'Gender' column, reset index
        x_train = x_train[train_mask].drop(columns=['Gender']).reset_index(drop=True)
        x_val = x_val[val_mask].drop(columns=['Gender']).reset_index(drop=True)
        x_test = x_test[test_mask].drop(columns=['Gender']).reset_index(drop=True)
        x_test_ext = x_test_ext[test_ext_mask].drop(columns=['Gender']).reset_index(drop=True)

        # Align y_*_m arrays with the filtered x_* DataFrames using the boolean mask
        y_train_m = y_train_m[train_mask.values]
        y_val_m = y_val_m[val_mask.values]
        y_test_m = y_test_m[test_mask.values]
        y_test_ext_m = y_test_ext_m[test_ext_mask.values]
    
    
    alpha = 0.05  # Define your alpha value for confidence interval

    # Train the model on the selected label
    y_train_i = y_train_m[:, idx_internal]
    y_val_i = y_val_m[:, idx_internal]
    y_test_i = y_test_m[:, idx_internal]
    y_test_ext_i = y_test_ext_m[:, idx_external]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')    
    
    # Train the model with early stopping
    model.fit(
        x_train, y_train_i,
        eval_set=[(x_val, y_val_i)],
        early_stopping_rounds=10,
        verbose=False
    )

    
   
    # Uncomment if want to investigate hyperparameters
    '''config = json.loads(model.get_booster().save_config())
    print(json.dumps(config, indent=4))'''
    
    
    
    # Generate predictions
    y_test_preds = model.predict_proba(x_test)[:, 1]
    y_test_ext_preds = model.predict_proba(x_test_ext)[:, 1]

    # Calculate AUCs
    base_auc = roc_auc_score(y_test_i, y_test_preds)
    external_auc = roc_auc_score(y_test_ext_i, y_test_ext_preds)



    # Uncomment this if want to investigate performance by demographic subgroups 
    '''
    # Compute AUROC for gender subgroups
    for gender in [0, 1]:
        mask_internal = x_test['Gender'] == gender
        mask_external = x_test_ext['Gender'] == gender

        auc_internal = roc_auc_score(y_test_i[mask_internal], y_test_preds[mask_internal])
        auc_external = roc_auc_score(y_test_ext_i[mask_external], y_test_ext_preds[mask_external])

        print(f"AUROC for Gender {gender} - Internal: {auc_internal:.4f}, External: {auc_external:.4f}")

    # Define age quantiles for internal and external datasets
    age_bins_internal = [18, 53, 66, 78, 101]
    age_bins_external = [18, 40, 52, 65, 109]

    for i in range(4):
        mask_internal = (x_test['Age'] >= age_bins_internal[i]) & (x_test['Age'] < age_bins_internal[i+1])
        mask_external = (x_test_ext['Age'] >= age_bins_external[i]) & (x_test_ext['Age'] < age_bins_external[i+1])

        auc_internal = roc_auc_score(y_test_i[mask_internal], y_test_preds[mask_internal])
        auc_external = roc_auc_score(y_test_ext_i[mask_external], y_test_ext_preds[mask_external])

        print(f"AUROC for Age Quantile {i+1} - Internal: {auc_internal:.4f}, External: {auc_external:.4f}")'''

    
    
    
    
    # Uncomment this if want to compare this with a baseline model which always makes negative predictions
    '''
    # Always-No Model (predicts all 0s)
    y_test_no = np.zeros_like(y_test_i)
    y_test_ext_no = np.zeros_like(y_test_ext_i)

    # AUC for Always-No Model
    base_auc_no = roc_auc_score(y_test_i, y_test_no)
    external_auc_no = roc_auc_score(y_test_ext_i, y_test_ext_no)

    # Sensitivity (Recall) for Always-No Model
    sensitivity_no = recall_score(y_test_i, y_test_no)
    sensitivity_no_ext = recall_score(y_test_ext_i, y_test_ext_no)

    # Specificity for Always-No Model (should be 1.0 since it predicts all negatives)
    specificity_no = 1.0
    specificity_no_ext = 1.0

    print(f"AUC of model (internal test set): {base_auc:.4f}")
    print(f"AUC of model (external test set): {external_auc:.4f}")
    print(f"AUC of always-no model (internal test set): {base_auc_no:.4f}")
    print(f"AUC of always-no model (external test set): {external_auc_no:.4f}")
    print(f"Sensitivity of always-no model (internal test set): {sensitivity_no:.4f}")
    print(f"Sensitivity of always-no model (external test set): {sensitivity_no_ext:.4f}")
    print(f"Specificity of always-no model (internal test set): {specificity_no:.4f}")
    print(f"Specificity of always-no model (external test set): {specificity_no_ext:.4f}")

    # ---- Find threshold for Sensitivity >= 80% ----
    def get_threshold_for_sensitivity(y_true, y_preds, target_sensitivity=0.7):
        fpr, tpr, thresholds = roc_curve(y_true, y_preds)
        idx = np.argmax(tpr >= target_sensitivity)  # Get first threshold meeting the condition
        return thresholds[idx]

    # Find optimal thresholds for internal and external sets
    threshold_internal = get_threshold_for_sensitivity(y_test_i, y_test_preds)
    threshold_external = get_threshold_for_sensitivity(y_test_ext_i, y_test_ext_preds)

    # Apply thresholds to get binary predictions
    y_test_binary = (y_test_preds >= threshold_internal).astype(int)
    y_test_ext_binary = (y_test_ext_preds >= threshold_external).astype(int)

    # Compute Sensitivity and Specificity
    sensitivity_test = recall_score(y_test_i, y_test_binary)
    sensitivity_ext = recall_score(y_test_ext_i, y_test_ext_binary)

    # Compute Specificity
    def compute_specificity(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)  # Specificity formula

    specificity_test = compute_specificity(y_test_i, y_test_binary)
    specificity_ext = compute_specificity(y_test_ext_i, y_test_ext_binary)

    print(f"Sensitivity-fixed (internal test set): {sensitivity_test:.4f}, Specificity: {specificity_test:.4f}")
    print(f"Sensitivity-fixed (external test set): {sensitivity_ext:.4f}, Specificity: {specificity_ext:.4f}")'''

    
    
    
    
    
    # Bootstrapping to calculate confidence intervals
    iterations = 1
    bootstrap_aucs_internal = []
    bootstrap_aucs_external = []

    for _ in tqdm(range(iterations)):
        # Generate bootstrap sample indices
        sample_indices = np.random.choice(len(y_test_i), len(y_test_i), replace=True)
        sample_indices_ext = np.random.choice(len(y_test_ext_i), len(y_test_ext_i), replace=True)

        # Internal test set bootstrapping
        y_test_bootstrap = y_test_i[sample_indices]
        y_test_preds_bootstrap = y_test_preds[sample_indices]
        if len(np.unique(y_test_bootstrap)) > 1:  # Ensure both classes are present
            auc_bootstrap_int = roc_auc_score(y_test_bootstrap, y_test_preds_bootstrap)
            bootstrap_aucs_internal.append(auc_bootstrap_int)

        # External test set bootstrapping
        y_test_ext_bootstrap = y_test_ext_i[sample_indices_ext]
        y_test_ext_preds_bootstrap = y_test_ext_preds[sample_indices_ext]
        if len(np.unique(y_test_ext_bootstrap)) > 1:  # Ensure both classes are present
            auc_bootstrap_ext = roc_auc_score(y_test_ext_bootstrap, y_test_ext_preds_bootstrap)
            bootstrap_aucs_external.append(auc_bootstrap_ext)

    # Compute confidence intervals
    bootstrap_aucs_internal = np.array(bootstrap_aucs_internal)
    auc_diff_int = bootstrap_aucs_internal - base_auc
    low_auc_int = base_auc + np.percentile(auc_diff_int, ((1.0 - alpha) / 2.0) * 100)
    high_auc_int = base_auc + np.percentile(auc_diff_int, (alpha + ((1.0 - alpha) / 2.0)) * 100)

    bootstrap_aucs_external = np.array(bootstrap_aucs_external)
    auc_diff_ext = bootstrap_aucs_external - external_auc
    low_auc_ext = external_auc + np.percentile(auc_diff_ext, ((1.0 - alpha) / 2.0) * 100)
    high_auc_ext = external_auc + np.percentile(auc_diff_ext, (alpha + ((1.0 - alpha) / 2.0)) * 100)
    
    
    prevalence_int = y_train_i.sum() + y_val_i.sum() + y_test_i.sum()
    prevalence_ext = y_test_ext_i.sum()
    
    prevalence_int = round((prevalence_int / (len(y_train_i) + len(y_val_i) + len(y_test_i)))*100,2)
    prevalence_ext = round((prevalence_ext / len(y_test_ext_m))*100,2)
    
    print('prevalence internal and external')
    print(prevalence_int, prevalence_ext)
    
    # Create a colormap object for RdBu
    cmap = plt.get_cmap("RdBu_r")

    # Define the normalization and scalar mappable object
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Select colors from the RdBu colormap for internal and external plots
    internal_color = sm.to_rgba(0.2)  # This gives a red shade (closer to 0 is more red)
    external_color = sm.to_rgba(0.8)  # This gives a blue shade (closer to 1 is more blue)

    # Plot AUROC curves
    fpr, tpr, _ = roc_curve(y_test_i, y_test_preds)
    fpr_ext, tpr_ext, _ = roc_curve(y_test_ext_i, y_test_ext_preds)

    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, color=internal_color, label=f'Int.: {base_auc:.4f} ({low_auc_int:.4f}, {high_auc_int:.4f}) [{prevalence_int}%]')
    plt.plot(fpr_ext, tpr_ext, color=external_color, linestyle='--', label=f'Ext.: {external_auc:.4f} ({low_auc_ext:.4f}, {high_auc_ext:.4f}) [{prevalence_ext}%]')
    plt.plot([0, 1], [0, 1], 'k-.', lw=2)  # Changing the random line to dot-dash
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'{code}: {description}', fontsize=16)
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'figures/{code}_auroc.png', dpi=600)
    plt.close()
    
    
    # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html
    # SHAP values plot using beeswarm
    explainer = shap.TreeExplainer(model, x_train)
    shap_values = explainer(x_train)

    # Ensure the SHAP plot is saved correctly
    fig, ax = plt.subplots(figsize=(4, 3))
    shap.plots.beeswarm(shap_values, color=plt.get_cmap("RdBu_r"), show=False)  # Use show=False to prevent auto-display
    
    plt.xlabel('SHAP Value (impact on model output)', fontsize=18)
    plt.ylabel('Features', fontsize=18)
    for label in ax.get_yticklabels():
        label.set_fontsize(14)  # Adjust fontsize as needed

    plt.title(f'{code}: {description}', fontsize=18)
    #plt.title(f'{code}', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figures/{code}_shap.png', dpi=600)
    plt.close()
    

    
    
features = ['sex', 'age', 'RR', 'PR', 'QRS', 'QT', 'QTc', 'P_wave_axis', 'QRS_axis', 'T_wave_axis']

df_internal = df_mimic
df_external = df_view

lbls_internal = lbls_mimic
lbls_external = lbls_view

# Separate train/val/test for internal dataset
df_train = df_internal[df_internal['strat_fold'].isin(range(0, 18))]
df_val = df_internal[df_internal['strat_fold'] == 18]
df_test = df_internal[df_internal['strat_fold'] == 19]

y_train = df_train['Diagnoses_labels']
y_val = df_val['Diagnoses_labels']
y_test = df_test['Diagnoses_labels']

y_train_m = np.stack(y_train.values)
y_val_m = np.stack(y_val.values)
y_test_m = np.stack(y_test.values)

x_train = df_train[features]
x_val = df_val[features]
x_test = df_test[features]

y_test_ext = df_external['Diagnoses_labels']
y_test_ext_m = np.stack(y_test_ext.values)
x_test_ext = df_external[features]

new_columns = ['Gender', 'Age', 'RR interval', 
               'PR interval', 'QRS duration', 'QT interval', 'QTc interval', 
               'P-wave axis', 'QRS axis', 'T-wave axis']


x_train.columns = new_columns
x_val.columns = new_columns
x_test.columns = new_columns

x_test_ext.columns = new_columns














####### NEOPLASMS

# Respiratory Conditions
# "C3490": "Malignant neoplasm of unspecified part of unspecified bronchus or lung"
respiratory_conditions = { "C343": "Malignant neoplasm of lower lobe, bronchus or lung", 
                          "C34": "Malignant neoplasm of bronchus and lung", 
                          "C341": "Malignant neoplasm of upper lobe, bronchus or lung",
                          "C349": "Malignant neoplasm of unspecified part of bronchus or lung"}


# Digestive System Conditions
# "C240": "Malignant neoplasm of extrahepatic bile duct" 
digestive_conditions = { "C15": "Malignant neoplasm of esophagus",
                        "C22": "Malignant neoplasm of liver and intrahepatic bile ducts",
                        "C24": "Malignant neoplasm of other and unspecified parts of biliary tract"}

# Cerebral Conditions
cerebral_conditions = {"C793": "Secondary malignant neoplasm of brain and cerebral meninges"}

# Urological Conditions (bladders both)
urological_conditions = { "C61": "Malignant neoplasm of prostate",
                         "N40": "Benign prostatic hyperplasia",
                         "N400": "Benign prostatic hyperplasia without lower urinary tract symptoms",
                         "C679": "Malignant neoplasm of bladder, unspecified"} 

# Gynecological Conditions (female only)
gynecological_conditions = { "D25": "Leiomyoma of uterus", 
                            "N80": "Endometriosis"}

for k,v in respiratory_conditions.items():
    get_plots(x_train, x_val, x_test,
                  y_train_m, y_val_m, y_test_m, 
                  x_test_ext, y_test_ext_m,
                  lbls_internal, lbls_external, 
                  features, 
                  k, v, 
                  keep_gender=None)

for k,v in digestive_conditions.items():
    get_plots(x_train, x_val, x_test,
                  y_train_m, y_val_m, y_test_m, 
                  x_test_ext, y_test_ext_m,
                  lbls_internal, lbls_external, 
                  features, 
                  k, v,
                  keep_gender=None)
    
for k,v in cerebral_conditions.items():
    get_plots(x_train, x_val, x_test,
                  y_train_m, y_val_m, y_test_m, 
                  x_test_ext, y_test_ext_m,
                  lbls_internal, lbls_external,
                  features,
                  k, v,
                  keep_gender=None)
    
for k,v in urological_conditions.items():
    if k=='C679':
        g = None
    else:
        g = 1
    print(v,g)
    get_plots(x_train, x_val, x_test,
                  y_train_m, y_val_m, y_test_m, 
                  x_test_ext, y_test_ext_m,
                  lbls_internal, lbls_external, 
                  features, 
                  k, v, 
                  keep_gender=g)

for k,v in gynecological_conditions.items():
    get_plots(x_train, x_val, x_test,
                  y_train_m, y_val_m, y_test_m, 
                  x_test_ext, y_test_ext_m,
                  lbls_internal, lbls_external, 
                  features, 
                  k, v, 
                  keep_gender=0)
    
    
    
    
    
    
    
####### NEUROPSYCHIATRIC

# Neurological Conditions 
neurological_conditions = { "G30": "Alzheimer's", # Alzheimer's disease 
                           "G20": "Parkinson's", # Parkinson's disease 
                           "G931": "Anoxic brain damage" } # Anoxic brain damage, not elsewhere classified 

# Psychiatric Conditions
psychiatric_conditions = { "F03": "Dementia",  # Unspecified dementia
                          "F01": "Vascular dementia",  # Vascular dementia
                          "F05": "Delirium due to physiological condition" } # Delirium due to physiological condition

for k,v in neurological_conditions.items():
    get_plots(x_train, x_val, x_test,
                  y_train_m, y_val_m, y_test_m, 
                  x_test_ext, y_test_ext_m,
                  lbls_internal, lbls_external, 
                  features, 
                  k, v, 
                  keep_gender=None)
    
for k,v in psychiatric_conditions.items():
    get_plots(x_train, x_val, x_test,
                  y_train_m, y_val_m, y_test_m, 
                  x_test_ext, y_test_ext_m,
                  lbls_internal, lbls_external, 
                  features,
                  k, v, 
                  keep_gender=None)
    
    
    
    
####### LIVER

liver_conditions = { "K70": "Alcoholic liver disease", 
                    "K703": "Alcoholic cirrhosis of liver", 
                    "K7030": "Alcoholic cirrhosis of liver without ascites", 
                    "K729": "Hepatic failure, unspecified", 
                    "K7290": "Hepatic failure, unspecified without coma", 
                    "K72": "Hepatic failure, not elsewhere classified" }

for k,v in liver_conditions.items():
    get_plots(x_train, x_val, x_test,
                  y_train_m, y_val_m, y_test_m, 
                  x_test_ext, y_test_ext_m,
                  lbls_internal, lbls_external, 
                  features, 
                  k, v, 
                  keep_gender=None)



import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def _cramers_v(confusion_matrix):
    """Calculate Cramer's V statistic for categorical-categorical association."""
    chi2, p, dof, ex = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if n>1:
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        min_corr=min((kcorr - 1), (rcorr - 1))
        if min_corr!=0:
            cramer_v = np.sqrt(phi2corr / min_corr)
        else:
            cramer_v = np.nan
    else: 
        cramer_v = np.nan
            
    return cramer_v, chi2, p, dof, ex

def _mcar_association_test(df):
    """Perform pairwise association test on a DataFrame with mixed categorical and non-categorical variables."""
    features = df.columns
    result_cramer_v = pd.DataFrame(index=features, columns=features)
    result_p= pd.DataFrame(index=features, columns=features)
    result_chi2=pd.DataFrame(index=features, columns=features)

    for i in range(len(features)):
        for j in range(len(features)):
            feature1 = features[i]
            feature2 = features[j]
            if feature1 !=feature2:
                confusion_matrix = pd.crosstab(df[feature1].notna().astype(int),df[feature2].fillna('missing'), dropna=False)

                cramers_tuple = _cramers_v(confusion_matrix)
                result_cramer_v.at[feature1, feature2] = cramers_tuple[0]
                result_p.at[feature1, feature2] = cramers_tuple[2]
                result_chi2.at[feature1, feature2] = cramers_tuple[1]

    return result_cramer_v, result_chi2, result_p

def list_mcar_missing_type(dataframe,mcar_threshold=0.8):
    "this function checks the association table and with a threshold tells the type of missing for that feature"
    association_result_pvalues=_mcar_association_test(dataframe)[2]
    missing_type_dict=dict()
    for feature in association_result_pvalues.columns:
        missing_type="mcar"
        for other_feature in association_result_pvalues.index:
            if other_feature != feature:
                if association_result_pvalues.at[feature,other_feature]<mcar_threshold:
                    missing_type="not_mcar"
        missing_type_dict[feature]=missing_type
    keys_with_mcar_value = [key for key, value in missing_type_dict.items() if value == 'mcar']

    return keys_with_mcar_value


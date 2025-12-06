import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA

def initialAssessment(datas, title = ""):

    metrics = []
    rows, cols = datas.shape
    
    na_rows = datas[datas.isna().any(axis=1)]
    na_cols = datas.columns[datas.isna().any()].tolist()

    metrics.append({
        "rows": rows,
        "columns": cols,
        "number rows with na": na_rows.shape[0],
        "number of columns with na":  len(na_cols)
    })

    table = pd.DataFrame(metrics)
    table = table.style.set_caption(f"Initial Assessment for {title}")
    display(table)

def generateEDATables(datas):
    types = set()
    levels = dict()

    number_rows = datas.shape[0]

    dataset = datas.copy()
    
    nominal_rows = []
    nominal_vals = []
    ordinal_rows = []
    ordinal_vals = []
    numerical_rows = []
    numerical_outlier_rows = []
    numerical_columns = []
    
    for col in dataset.columns:
        c = dataset[col].dropna().values # use values to leverage np over pd
        v = dataset[col].value_counts(dropna = True)
        types.add(c.dtype)
        missing_v = dataset[col].isna().sum()
    
        if len(c) == 0 or c.dtype == "O":
            levels[col] = "nominal" # empty or probably a string...probably
            print(f"Column {col} is {levels[col]} with length: {len(c)}")
            
            nominal_rows.append({
                "column": col,
                "level": levels[col],
                "n": len(c),
                "missing values": missing_v,
                "missing %": missing_v / float(len(c)) * 100 ,
                "unique": len(np.unique(c)),
                "mode": v.index[0],
                "mode_n": v.iloc[0]
            })

            le = LabelEncoder()
            ce = le.fit_transform(c)
            uniq = np.unique(ce)
            count = []
            labely = []
            for cl in uniq:
                labely.append(cl)
                count.append(ce[ce == cl].shape[0])

            labely = le.inverse_transform(labely)
                
            plt.figure(figsize = (12, 8))
            plt.bar(labely, count)
            plt.grid(True)
            plt.show()

            cats = dict()
            cats["Total"] = sum(count)
            for cl in le.fit_transform(labely):
                cats[f"{le.inverse_transform([cl])} Count"] = ce[ce == cl].shape[0]
                cats[f"{le.inverse_transform([cl])} Proportion"] = ce[ce == cl].shape[0]/sum(count) * 100
            
            nominal_vals.append(cats)
            nominal_table_c = pd.DataFrame(nominal_vals)
            nominal_table_c = nominal_table_c.style.set_caption("Categorical Distribution")
            display(nominal_table_c)
            
            nominal_vals.append({f"{col}": np.unique(c)})
    
        else:
    
            intish = np.all(np.isclose(c, np.round(c)))
            uniques = len(np.unique(c))
            c_min, c_max = np.min(c), np.max(c)
            hasNegatives = (c < 0).any()
    
            if intish and uniques < 10: # if the number of uniques is small relative to length of column, might represent classes and not numbers
                levels[col] = "ordinal"

                le = LabelEncoder()
                ce = le.fit_transform(c)
                uniq = np.unique(ce)
                count = []
                labely = []
                for cl in uniq:
                    labely.append(cl)
                    count.append(ce[ce == cl].shape[0])
    
                labely = le.inverse_transform(labely)
                    
                plt.figure(figsize = (12, 8))
                plt.bar(labely, count)
                plt.grid(True)
                plt.show()

                cats = dict()
                cats["Total"] = sum(count)
                for cl in le.fit_transform(labely):
                    cats[f"{le.inverse_transform([cl])} Count"] = ce[ce == cl].shape[0]
                    cats[f"{le.inverse_transform([cl])} Proportion"] = ce[ce == cl].shape[0]/sum(count) * 100

                ordinal_vals.append(cats)
                ordinal_table_c = pd.DataFrame(ordinal_vals)
                ordinal_table_c = ordinal_table_c.style.set_caption("Categorical Distribution")
                display(ordinal_table_c)
                
                ordinal_vals.append({f"{col}": np.unique(c)})
    
                if pd.api.types.is_numeric_dtype(c):
                    ordinal_rows.append({
                    "column": col,
                    "level": levels[col],
                    "n": len(c),
                    "missing values": missing_v,
                    "missing %": missing_v / float(len(c)) * 100,
                    "unique": len(np.unique(c)),
                    "mode_n": v.iloc[0],
                    "mode": v.index[0],
                    "min": c_min,
                    "max": c_max,
                    "mean": np.mean(c),
                    "SD": np.std(c)
                    })
                else:
                    ordinal_rows.append({
                    "column": col,
                    "level": levels[col],
                    "n": len(c),
                    "missing values": missing_v,
                    "missing %": missing_v / float(len(c)) * 100,
                    "unique": len(np.unique(c)),
                    "mode_n": v.iloc[0],
                    "mode": v.index[0],
                    "min": c_min,
                    "max": c_max,
                    "mean": np.mean(c),
                    "SD": np.std(c),
                    "Q1": dataset[col].quantile(.25),
                    "Q2 (Median)": dataset[col].quantile(.50),
                    "Q3": dataset[col].quantile(.75),
                    "IQR": dataset[col].quantile(.75) - dataset[col].quantile(.25)
                    })
                
            else:
                if hasNegatives:
                    levels[col] = "interval"
                else:
                    levels[col] = "ratio"

                numerical_columns.append(col)
    
                numerical_rows.append({
                    "column": col,
                    "level": levels[col],
                    "n": len(c),
                    "missing values": missing_v,
                    "missing %": missing_v / float(len(c)) * 100,
                    "unique": len(np.unique(c)),
                    "mode_n": v.iloc[0],
                    "mode": v.index[0],
                    "min": c_min,
                    "max": c_max,
                    "skew": dataset[col].skew(skipna = True),
                    "kurtosis": dataset[col].kurtosis(skipna = True)
                    })

                iqr = dataset[col].quantile(.75) - dataset[col].quantile(.25)
                lower_quartile_outliers = (c < (dataset[col].quantile(.25) - (1.5 * iqr))).sum()
                upper_quartile_outliers = (c > (dataset[col].quantile(.75) + (1.5 * iqr))).sum()

                mean = np.mean(c)
                std = np.std(c)
                lower_std_outliers = (c < (np.mean(c) - (3 * np.std(c)))).sum()
                upper_std_outliers = (c > (np.mean(c) + (3 * np.std(c)))).sum()
                
                left_outliers = lower_quartile_outliers + lower_std_outliers
                right_outliers = upper_quartile_outliers + upper_std_outliers
                
                numerical_outlier_rows.append({
                    "column": col,
                    "Q1": dataset[col].quantile(.25),
                    "Q2 (Median)": dataset[col].quantile(.50),
                    "Q3": dataset[col].quantile(.75),
                    "IQR": iqr,
                    "n < (Q1 - 1.5 * IQR)": lower_quartile_outliers,
                    "n > (Q3 + 1.5 * IQR)": upper_quartile_outliers,
                    "mean": mean,
                    "SD": std,
                    "n < (mean - (3 * SD))": lower_std_outliers,
                    "n > (mean + (3 * SD))": upper_std_outliers,
                    "Possible Left Outliers": left_outliers,
                    "Possible Right Outliers": right_outliers,
                    "Possible Outliers": left_outliers + right_outliers,
                    "Outlier Percentage": (left_outliers + right_outliers) / number_rows * 100
                })

                #cdata = data['p03'].values
                #hcdata = cdata[(cdata > data['p03'].quantile(.75))]
                #print("threshold: ", data['p03'].quantile(.75))
                #hcdata.size

    if len(nominal_rows) != 0:
        table = pd.DataFrame(nominal_rows)
        table = table.style.set_caption("Nominal Columns")
        display(table)
    if len(ordinal_rows) != 0:
        table = pd.DataFrame(ordinal_rows)
        table = table.style.set_caption("Ordinal Columns")
        display(table)
    if len(numerical_rows) != 0:
        table = pd.DataFrame(numerical_rows)
        table = table.style.set_caption("Numerical Columns")
        display(table)
    if len(numerical_outlier_rows) != 0:
        table = pd.DataFrame(numerical_outlier_rows)
        table = table.style.set_caption("Numerical Outlier Columns")
        display(table)

        print("Numerical columns list\n", numerical_columns)

def profileFeatures(dataset, ycolumn = None):

    num_cols = [c for c in dataset.columns if pd.api.types.is_numeric_dtype(dataset[c])]
    cat_cols = [c for c in dataset.columns if not pd.api.types.is_numeric_dtype(dataset[c])]
    _, axs = plt.subplots(len(num_cols) + len(cat_cols), 3, figsize=(15, 5 * (len(num_cols) + len(cat_cols))))
    x = np.arange(len(dataset))

    row = 0
    
    for col in num_cols:

        y = dataset[col]
        yd = y.dropna()
        m = ~y.isna()
    
        if ycolumn == None or col == ycolumn:
           
            axs[row, 0].scatter(x[m], y[m], alpha = .6)
            axs[row, 0].grid(axis = "both", linestyle = '--', alpha = .5)
            axs[row, 0].set_facecolor("honeydew")
            
        else:
            
            axs[row, 0].scatter(y[m], dataset[ycolumn][m], alpha = .6)
            axs[row, 0].grid(axis = "both", linestyle = '--', alpha = .5)
            axs[row, 0].set_xlabel(col)
            axs[row, 0].set_ylabel(ycolumn)
            axs[row, 0].set_title(f"{col} vs {ycolumn}")
            axs[row, 0].set_facecolor("honeydew")
        
        axs[row, 1].hist(yd, bins = 50, density = True, alpha = .5, label = "hist")
        yd.plot(kind = "kde", ax = axs[row, 1], label = "kde")
        axs[row, 1].set_xlabel(col)
        axs[row, 1].set_ylabel("density")
        axs[row, 1].set_title(f"{col}: Histogram + KDE")
        axs[row, 1].grid(axis = "both", linestyle = '--', alpha = .5)
        axs[row, 1].set_facecolor("honeydew")
        axs[row, 1].legend()
        
        axs[row, 2].boxplot(yd, vert = True, patch_artist = True, boxprops = dict(facecolor = 'lightblue', color = 'navy'), medianprops = dict(color = 'red', linewidth = 1.5))
        axs[row, 2].set_title(f"Box Plot for {col}")
        axs[row, 2].grid(axis = "both", linestyle = '--', alpha = .5)
        axs[row, 2].set_facecolor("honeydew")


        row += 1

    for col in cat_cols:

        y = dataset[col]
        v = y.value_counts(dropna = False)
        
        if not v.empty:
            
            v.plot(kind = "bar", ax = axs[row, 0])
            axs[row, 0].set_ylabel("frequency")
            axs[row, 0].set_title("categorical frequency")
            axs[row, 0].set_facecolor("honeydew")

            row += 1

    plt.figure(figsize = (15, 8))
    dataset.boxplot(column = num_cols)

    plt.tight_layout()
    plt.show()

def profileIntercorrelation(datas):

    dataset = datas.copy()
    numerical_dataset = dataset.select_dtypes(include = ['number'])
    corr = numerical_dataset.corr()
    
    plt.figure(figsize = (12, 10))
    sns.heatmap(corr, cmap = 'coolwarm', annot = False, square = True)
    plt.title ("Correlation Heatmap")
    plt.show()
    

    correlations = []
    for col in corr.columns:
        discard = corr[col].index.isin([col])
        column = corr[col][~discard]
        correlations.append({
            "feature": col,
            "max correlation": column.max(),
            "max with": column.idxmax(),
            "min correlation": column.min(),
            "min with": column.idxmin()
            })

    if len(correlations) > 0:
        table = pd.DataFrame(correlations)
        table = table.style.set_caption("Correlations")
        display(table)

    numerical_dataset = add_constant(numerical_dataset)

    vif_dataset = pd.DataFrame()
    vif_dataset["Features"] = numerical_dataset.columns
    vif_dataset["VIF"] = [variance_inflation_factor(numerical_dataset.values, i) for i in range(numerical_dataset.shape[1])]
    vif_dataset = vif_dataset.sort_values(by='VIF', ascending=False)
    
    display(vif_dataset)

    return vif_dataset

def boxPlotData(datas):
    # normalize numerical columns

    data_n = datas.copy()
    
    num_cols = [c for c in data_n.columns if pd.api.types.is_numeric_dtype(data_n[c])]
    
    plt.figure(figsize = (14, 10))
    sns.boxplot(data = data_n[num_cols], palette = 'viridis')
    plt.title("Boxplot", fontsize = 20)
    plt.ylabel("Scale")
    plt.xlabel("Numerical Features")
    plt.show()

def probeVifRemoval(datas, threshold = 5.0):

    dataset = datas.copy()
    numerical_dataset = dataset.select_dtypes(include = ['number'])

    ordinal_vifs = []
    vifs = []

    iterate = True

    while iterate == True:
        
        vif_dataset = pd.DataFrame()
        vif_dataset["Features"] = numerical_dataset.columns
        vif_dataset["VIF"] = [variance_inflation_factor(numerical_dataset.values, i) for i in range(numerical_dataset.shape[1])]
        vif_dataset = vif_dataset.sort_values(by='VIF', ascending=False)
    
        highest_vif_feature, highest_vif_value = vif_dataset["Features"].iloc[0], vif_dataset["VIF"].iloc[0]

        if highest_vif_value >= threshold:
            numerical_dataset = numerical_dataset.drop(columns = [highest_vif_feature])
            ordinal_vifs.append({
                "Feature": highest_vif_feature,
                "VIF": highest_vif_value
            })
            vifs.append(highest_vif_feature)

        else:
            iterate = False

    print("Recommend Removing Following Features after Iteratively performing VIF Analysis")
    table_vif = pd.DataFrame(ordinal_vifs)
    table_vif = table_vif.style.set_caption("VIFS in Order")
    display(table_vif)

    return vifs

def performPcaAnalysis(datas, text_padding = 1.0):

    x = datas.copy()

    pca = PCA(n_components = len(x.columns))

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    x_pca = pca.fit_transform(x)
    cs = pca.explained_variance_ratio_.cumsum()

    plt.figure(figsize = (12, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio per Principal Component')

    meline_80 = np.argmax(cs >= .8) + (np.max(pca.explained_variance_ratio_) * 1.2)
    meline_90 = np.argmax(cs >= .9) + (np.max(pca.explained_variance_ratio_) * 1.2)
    meline_95 = np.argmax(cs >= .95) + (np.max(pca.explained_variance_ratio_) * 1.2)

    meline_800 = np.argmax(cs >= .8) + 1
    meline_900 = np.argmax(cs >= .9) + 1
    meline_950 = np.argmax(cs >= .95) + 1
    
    plt.axvline(meline_80, color = 'green', linestyle = '--', linewidth = 2)
    plt.axvline(meline_90, color = 'yellow', linestyle = '--', linewidth = 2)
    plt.axvline(meline_95, color = 'red', linestyle = '--', linewidth = 2)
    plt.text(meline_80 + text_padding, (np.max(pca.explained_variance_ratio_) * .5), "80% boundary")
    plt.text(meline_90 + text_padding, (np.max(pca.explained_variance_ratio_) * .65), "90% boundary")
    plt.text(meline_95 + text_padding, (np.max(pca.explained_variance_ratio_) * .8), "95% boundary")
    #plt.grid(True)
    
    plt.show()

    plt.figure(figsize = (12, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio per Principal Component')
    #plt.grid(True)
    
    plt.plot(range(1, len(cs) + 1), cs, marker='*', linestyle='-', c = 'r',
             label='Cumulative Explained Variance Ratio')
    plt.axvline(meline_800, color = 'green', linestyle = '--', linewidth = 2)
    plt.axvline(meline_900, color = 'yellow', linestyle = '--', linewidth = 2)
    plt.axvline(meline_950, color = 'red', linestyle = '--', linewidth = 2)
    plt.text(meline_800 + text_padding, .5, "80% boundary")
    plt.text(meline_900 + text_padding, .65, "90% boundary")
    plt.text(meline_950 + text_padding, .8, "95% boundary")
    plt.legend()
    
    plt.show()

    plt.figure(figsize = (12, 6))
    plt.scatter(x_pca[: ,0], x_pca[: ,1], c = "blue", edgecolor = "k")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PC1 vs PC2')
    plt.grid(True)
    
    plt.show()
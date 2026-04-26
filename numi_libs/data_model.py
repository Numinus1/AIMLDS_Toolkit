import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import (mean_squared_error, mean_absolute_error, root_mean_squared_error, 
    r2_score, accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, silhouette_score, davies_bouldin_score)
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE

import statsmodels.api as sm

seed = 420

def applyAndReportRegressors(x_tr, x_te, y_tr, y_te, algorithms, pca = [], visualizeByFeature = False, title = "Vanilla"):

    if len(algorithms) == 0:
        return None, None

    metrics_tr = []
    metrics_te = []

    if len(pca) == 0:
        pca.append(0)

    for component_count in pca:

        if component_count != 0:
                pca = PCA(n_components = component_count)
                x_tr = pca.fit_transform(x_tr)
                x_te = pca.transform(x_te)

        for name, algorithm in algorithms.items():
            
            algorithm.fit(x_tr, y_tr)
    
            y_tr_pr = algorithm.predict(x_tr)
            y_te_pr = algorithm.predict(x_te)

            y_tr_res = y_tr - y_tr_pr
            y_te_res = y_te - y_te_pr
    
            # Training Scores

            # Adjusted R Squared = 1 - [ (1 - R_Squared)(N - 1) ]/[ N - p - 1 ]
            # - N = Size of Sample
            # - p = number of variables
            # - R Squared = 1 - (RSS)/(TSS)
            # - RSS = Residual Sum of Squares; sum[ (y_i - y_est)^2 ]
            # - TSS = Total Sum of Squares; sum[ (y_i - y_avg)^2 ]
            # - [0, 1] - higher = better
            r2_tr = algorithm.score(x_tr, y_tr)
            rmse_tr = root_mean_squared_error(y_tr, y_tr_pr)
    
            res_tr = y_tr - y_tr_pr
    
            dbwat_tr = sm.stats.durbin_watson(res_tr)
            jb_tr, jb_p_tr, _, _ = sm.stats.jarque_bera(res_tr)
    
            # Testing Scores
    
            r2_te = algorithm.score(x_te, y_te)
            rmse_te = root_mean_squared_error(y_te, y_te_pr)
    
            res_te = y_te - y_te_pr
    
            dbwat_te = sm.stats.durbin_watson(res_te)
            jb_te, jb_p_te, _, _ = sm.stats.jarque_bera(res_te)

            metrics_te.append({
                "Algorithm": name,
                "PCA n": int(component_count),
                "R-Squared": r2_te,
                "RMSE": rmse_te,
                "Durbin-Watson": dbwat_te,
                "Jarque-Bera": jb_te,
                "Jarque_bera P-Value": jb_p_te
            })

            metrics_tr.append({
                "Algorithm": name,
                "PCA n": int(component_count),
                "R-Squared": r2_tr,
                "RMSE": rmse_tr,
                "Durbin-Watson": dbwat_tr,
                "Jarque-Bera": jb_tr,
                "Jarque_bera P-Value": jb_p_tr
            })

            if visualizeByFeature == True:
                for fe in x_tr.columns:
                    _, axs = plt.subplots(1, 2, figsize=(20, 5))
    
                    axs[0].scatter(x_tr[fe], y_tr, marker = 'o', c = 'b', label = "training data")
                    axs[0].scatter(x_te[fe], y_te, marker = '*', c = 'r', label = "predicted data")
                    axs[0].legend()
                    axs[0].set_xlabel(f"{fe}")
                    axs[0].set_ylabel("Y")
                    axs[0].set_title(f"{title} - {name} - {fe} Feature vs Output")
                    
                    axs[1].scatter(x_tr[fe], y_tr_res, marker = 'o', c = 'b', label = "training data")
                    axs[1].scatter(x_te[fe], y_te_res, marker = '*', c = 'g', label = "testing data")
                    axs[1].set_xlabel(f"{fe}")
                    axs[1].set_ylabel("Residuals")
                    axs[1].axhline(y = 0, linestyle = '--')
                    axs[1].set_title(f"{title} - {name} - {fe} - Residuals Plot")
        
                plt.tight_layout()
                plt.show()

            _, axs = plt.subplots(1, 2, figsize=(20, 5))
            
            axs[0].hist(y_te_res, bins = 50)
            axs[0].set_xlabel("Residuals - Testing")
            axs[0].set_title(f"PCA {component_count} - Residual Distribution")
            
            sm.qqplot(y_te_res, line = "45", ax = axs[1], fit = True)
            axs[1].set_title(f"PCA {component_count} - QQ Plot - Residuals Training")

            plt.tight_layout()
            plt.show()

    table_te = pd.DataFrame(metrics_te)
    table_te = table_te.sort_values(by=["R-Squared", "RMSE"], ascending = [False, True])
    table_te = table_te.style.set_caption(f"{title} - Testing Data")
    display(table_te)
    
    table_tr = pd.DataFrame(metrics_tr)
    table_tr = table_tr.sort_values(by=["R-Squared", "RMSE"], ascending = [False, True])
    table_tr = table_tr.style.set_caption(f"{title} - Training Data")
    display(table_tr)

def applyAndReportClassifiers(x_tr, x_te, y_tr, y_te, algorithms, pca = [], title = "", path_to_persist_predictions = None):

    metric_tables = []
    metric_tables_per_class = []
    metric_tables_avg = []

    #org_tr = x_tr[['file', 'cell_n']]
    #org_te = x_te[['file', 'cell_n']]
    #display(org_te)

    #x_tr = x_tr.drop(columns = ['file', 'cell_n'])
    #x_te = x_te.drop(columns = ['file', 'cell_n'])
    
    le = LabelEncoder()
    y_combined = pd.concat([y_tr, y_te])
    y_combined = le.fit_transform(y_combined)

    y_tr = le.transform(y_tr)
    y_te = le.transform(y_te)
    classes = np.unique([*y_combined])

    if len(pca) == 0:
        pca.append(0)

    for name, algo in algorithms.items():
            
        for component_count in pca:
        
            if component_count != 0:
                pca = PCA(n_components = component_count)
                x_tr = pca.fit_transform(x_tr)
                x_te = pca.transform(x_te)
  
            algo.fit(x_tr, y_tr)
    
            y_tr_pr = algo.predict(x_tr)
            y_te_pr = algo.predict(x_te)

            if path_to_persist_predictions != None:

                y_te_pr_ex = le.inverse_transform(y_te_pr)

                pred_data = org_te.copy()
                pred_data["classification"] = le.inverse_transform(y_te)
                pred_data["prediction"] = y_te_pr_ex

                pred_data.to_csv(path_to_persist_predictions, index = False)

            # Accuracy = (Correct Predictions)/(Total Predictions)
            # - Misleading if Data Imbalanced - can be high accuracy for majority but not minorty
            matrix_tr = confusion_matrix(y_tr, y_tr_pr)
            acc_tr = matrix_tr.diagonal()/matrix_tr.sum(axis=1)

            matrix_te = confusion_matrix(y_te, y_te_pr)
            acc_te = matrix_te.diagonal()/matrix_te.sum(axis=1)

            # Precision = TP / (TP + FP)
            # - Useful when cost of FP is very high
            prec_tr = precision_score(y_tr, y_tr_pr, average = None, labels = classes, zero_division=0)
            prec_te = precision_score(y_te, y_te_pr, average = None, labels = classes, zero_division=0)

            # Recall = TP / (TP + FN)
            # - Useful when cost of FN > cost of FP
            rec_tr = recall_score(y_tr, y_tr_pr, average = None, labels = classes, zero_division=0)
            rec_te = recall_score(y_te, y_te_pr, average = None, labels = classes, zero_division=0)

            # F1 = 2 x [ (Precision x Recall) / (Precision + Recall) ]
            # - Harmonic Mean of Precision & Recall
            # - A balanced measure - always useful
            f1_tr = f1_score(y_tr, y_tr_pr, average = None, labels = classes, zero_division=0)
            f1_te = f1_score(y_te, y_te_pr, average = None, labels = classes, zero_division=0)
            
            y_tr_prb = algo.predict_proba(x_tr)
            y_te_prb = algo.predict_proba(x_te)

            auc_tr = None
            auc_te = None

            if len(classes) > 2:
                auc_tr = roc_auc_score(y_tr, y_tr_prb, multi_class = "ovr", average = "macro")
                auc_te = roc_auc_score(y_te, y_te_prb, multi_class = "ovr", average = "macro")
            else:
                auc_tr = roc_auc_score(y_tr, y_tr_prb[:, 1], multi_class = "ovr", average = "macro")
                auc_te = roc_auc_score(y_te, y_te_prb[:, 1], multi_class = "ovr", average = "macro")

            auc_tr_class = dict()
            auc_te_class = dict()

            acc_tr_avg = 0.0
            prec_tr_avg = 0.0
            rec_tr_avg = 0.0
            f1_tr_avg = 0.0
            auc_tr_avg = 0.0

            acc_te_avg = 0.0
            prec_te_avg = 0.0
            rec_te_avg = 0.0
            f1_te_avg = 0.0
            auc_te_avg = 0.0

            plt.figure(figsize = (15, 10))
            displayMatrix = ConfusionMatrixDisplay(confusion_matrix = matrix_te, display_labels = le.inverse_transform(classes))
            displayMatrix.plot(cmap = 'Greens')
            plt.title(f"{title} - Testing Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.show()

            plt.figure(figsize = (15, 10))
            plt.title(f"{title} - ROC - AUC Curve - Testing")
            plt.ylabel("FP - Specificity")
            plt.xlabel("TP - Sensitivity")

            for i, c in enumerate(classes):
                
                y_tr_bin = (y_tr == c).astype(int)
                y_te_bin = (y_te == c).astype(int)

                y_tr_prb_bin = y_tr_prb[:, i]
                y_te_prb_bin = y_te_prb[:, i]

                auc_tr_c = roc_auc_score(y_tr_bin, y_tr_prb_bin)
                auc_te_c = roc_auc_score(y_te_bin, y_te_prb_bin)

                fp, tp, _ = roc_curve(y_te_bin, y_te_prb_bin)

                class_name = le.inverse_transform([c])[0]

                plt.plot(fp, tp, label = f"ROC - {class_name}: {auc_te_c:.2f}")

                metric_tables_per_class.append(
                {
                    "Algorithm": name,
                    "PCA N": component_count,
                    "Class": class_name,
                    
                    "F1 Score - Testing": f1_te[i],
                    "AUC - Testing": auc_te_c,
                    "Accuracy - Testing": acc_te[i],
                    "Precision - Testing": prec_te[i],
                    "Recall - Testing": rec_te[i],
                    
                    "F1 Score - Training": f1_tr[i],
                    "AUC - Training": auc_tr_c,
                    "Accuracy - Training": acc_tr[i],
                    "Precision - Training": prec_tr[i],
                    "Recall - Training": rec_tr[i]
                })

                acc_tr_avg += acc_tr[i]/len(classes)
                prec_tr_avg += prec_tr[i]/len(classes)
                rec_tr_avg += rec_tr[i]/len(classes)
                f1_tr_avg += f1_tr[i]/len(classes)
                auc_tr_avg += auc_tr_c/len(classes)

                acc_te_avg += acc_te[i]/len(classes)
                prec_te_avg += prec_te[i]/len(classes)
                rec_te_avg += rec_te[i]/len(classes)
                f1_te_avg += f1_te[i]/len(classes)
                auc_te_avg += auc_te_c/len(classes)

            plt.plot([0, 1], [0, 1],'r--')
            plt.legend()
            plt.show()

            metric_tables_avg.append(
                {
                    "Algorithm": name,
                    "PCA N": component_count,
                    
                    "F1 Score - Testing Avg": f1_te_avg,
                    "AUC - Testing Avg": auc_te_avg,
                    "Accuracy - Testing Avg": acc_te_avg,
                    "Precision - Testing Avg": prec_te_avg,
                    "Recall - Testing Avg": rec_te_avg,
                    
                    "F1 Score - Training Avg": f1_tr_avg,
                    "AUC - Training Avg": auc_tr_avg,
                    "Accuracy - Training Avg": acc_tr_avg,
                    "Precision - Training Avg": prec_tr_avg,
                    "Recall - Training Avg": rec_tr_avg
                })
    
            acc_tr = accuracy_score(y_tr, y_tr_pr)
            prec_tr = precision_score(y_tr, y_tr_pr, average='weighted', zero_division=0)
            rec_tr = recall_score(y_tr, y_tr_pr, average='weighted', zero_division=0)
            f1_tr = f1_score(y_tr, y_tr_pr, average='weighted', zero_division=0)
    
            acc_te = accuracy_score(y_te, y_te_pr)
            prec_te = precision_score(y_te, y_te_pr, average='weighted', zero_division=0)
            rec_te = recall_score(y_te, y_te_pr, average='weighted', zero_division=0)
            f1_te = f1_score(y_te, y_te_pr, average='weighted', zero_division=0)
    
            metric_tables.append(
                {
                    "Algorithm": name,
                    "PCA N": component_count,
                    
                    "F1 Score - Testing": f1_te,
                    "Accuracy - Testing": acc_te,
                    "Precision - Testing": prec_te,
                    "Recall - Testing": rec_te,
                    
                    "F1 Score - Training": f1_tr,
                    "Accuracy - Training": acc_tr,
                    "Precision - Training": prec_tr,
                    "Recall - Training": rec_tr
                }
            )

    if metric_tables != None:
        table = pd.DataFrame(metric_tables)
        table = table.style.set_caption(f"{title} - Scores")
        display(table)

    if metric_tables_per_class != None:
        table = pd.DataFrame(metric_tables_per_class)
        table = table.style.set_caption(f"{title} - Scores by Class")
        display(table)
    
    if metric_tables_avg != None:
        table = pd.DataFrame(metric_tables_avg)
        table = table.style.set_caption(f"{title} - Scores by Class Avg")
        display(table)

def performTSNE(datas, components = 2):
    
    x = datas.copy()
    
    tsne = TSNE(n_components = components, random_state = seed)
    x_e = tsne.fit_transform(x)

    _, axs = plt.subplots(1, 2, figsize=(15, 6))

    axs[0].scatter(x_e[: ,0], x_e[: ,1], c = "blue", edgecolor = "k")
    axs[0].set_xlabel('PC1')
    axs[0].set_ylabel('PC2')
    axs[0].set_title(f'PC1 vs PC2 - Vanilla n = {components}')
    axs[0].grid(axis = "both", linestyle = '--', alpha = .5)
    axs[0].set_facecolor("honeydew")

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    tsne = TSNE(n_components = components, random_state = seed)
    x_e = tsne.fit_transform(x)

    axs[1].scatter(x_e[: ,0], x_e[: ,1], c = "blue", edgecolor = "k")
    axs[1].set_xlabel('PC1')
    axs[1].set_ylabel('PC2')
    axs[1].set_title(f'PC1 vs PC2 - Scaled n = {components}')
    axs[1].grid(axis = "both", linestyle = '--', alpha = .5)
    axs[1].set_facecolor("honeydew")
    
    plt.show()
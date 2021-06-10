from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import numpy as np
import pandas as pd

ORDER = ['conflicto', 'economico', 'humanidad', 'moral']
COLUMNS = ['original_text', 'preprocess_text', 'encoded', 'frames', 'conflicto', 'economico', 'humanidad', 'moral']


def load_dataset(filename):
    df = np.load("datasets/" + filename, allow_pickle=True)
    df = pd.DataFrame(df, columns=COLUMNS)
    return df



def predict(model, data, y_true, mode, model_name, step):
    try:
        frames_probability = model.predict_proba(data)
        frames_probability = [[f_prob[1] for f_prob in frames] for frames in zip(*frames_probability)]
        y_pred = [[int(pred >= 0.5) for pred in frames] for frames in frames_probability]
        
    except:
        y_pred = model.predict(data).tolist()
        frames_probability = y_pred.tolist()
        
    df_result = pd.DataFrame()
    df_result["y_pred"] = y_pred
    df_result["y_prob"] = frames_probability
    df_result["y_true"] = y_true.tolist()
    
    df_result.to_pickle(f"Results/{model_name}#{mode}#{step}.pkl")
    data = process_folds([df_result])
    
    print("\n" + step + "\n")
    build_report(pd.DataFrame(data.mean()).T, data.applymap(lambda x:0), mode)
    
    return df_result


def predict_deep(model, data, y_true, mode, model_name, step):
    frames_probability = model.predict_proba(data).tolist()
    y_pred = [[int(pred >= 0.5) for pred in frames] for frames in frames_probability]
        
    
    df_result = pd.DataFrame()
    df_result["y_pred"] = y_pred
    df_result["y_prob"] = frames_probability
    df_result["y_true"] = y_true.tolist()
    
    df_result.to_pickle(f"Results/{model_name}#{mode}#{step}.pkl")

    data = process_folds([df_result])
    
    print("\n" + step + "\n")
    build_report(pd.DataFrame(data.mean()).T, data.applymap(lambda x:0), mode)
    
    return df_result


def load_embedding(mode, dataset):
    if mode not in ["beto_embedding_cls", "beto_embedding_mean", "elmo", "fasttext", "fasttext_512"]:
        raise ValueError()
        
    if mode in ["elmo", "fasttext"]:
        vector = np.load(f"datasets/{mode}_{dataset}.npz", allow_pickle=True)["arr_0"]
    else:
        vector = np.load(f"datasets/{mode}_{dataset}.npy", allow_pickle=True)
    return vector


def process_folds(datasets, order=ORDER):
    final_dataset = []
    for dataset in datasets:
        data = {}
        def map_array(x):
            string = " "
            if "," in x:
                string = ","
            return [int(i) for i in x.strip("()").strip("[]").split(string)]
        
        def map_y_pred_prob(x):
            return [float(i) for i in x.strip("[]").split(",")]

        y_true = dataset['y_true'].tolist()
        y_pred = dataset['y_pred'].tolist()
        y_pred_prob = dataset['y_prob'].tolist()
        for index, frame in enumerate(order):

            y_true_frame = [x[index] for x in y_true]
            y_pred_frame = [x[index] for x in y_pred]
            y_pred_prob_frame = [x[index] for x in y_pred_prob]
            report = classification_report(y_true_frame, y_pred_frame, output_dict=True, zero_division=False)
            report_2 = classification_report(y_true_frame, y_pred_frame, output_dict=False, zero_division=False)
            

            for class_ in ["0", "1"]:
                for key in report[class_]:
                    data[f"{frame}_{class_}_{key}"]=float(report[class_][key])

            for measure in ["macro avg", "weighted avg"]:
                for key in report[measure]:
                    data[f"{frame}_{measure}_{key}"]=float(report[measure][key])
                    
            data[f"{frame}_acc"] = float(report["accuracy"])

            precision, recall, _ = precision_recall_curve(y_true_frame, y_pred_prob_frame)

            data[f"{frame}_auc_press_recall"] = auc(recall, precision)
            data[f"{frame}_roc_auc_score"] = roc_auc_score(y_true_frame, y_pred_prob_frame)

        final_dataset.append(data)

    return pd.DataFrame.from_dict(final_dataset)


def build_report(data, data_std, mode, order=ORDER):
    all_recall_micro = []
    all_recall_macro = []
    all_pre_micro = []
    all_pre_macro = []
    all_f1_micro = []
    all_f1_macro = []
    all_acc = []
    
    all_auc = []
    all_std_auc = []
    all_roc = []
    all_std_roc = []
    
    all_std_recall_micro = []
    all_std_recall_macro = []
    all_std_pre_micro = []
    all_std_pre_macro = []
    all_std_f1_micro = []
    all_std_f1_macro = []

    for frame in order:

        pres_0 = data[f'{frame}_0_precision'][0]
        pres_1 = data[f'{frame}_1_precision'][0]
        recall_0 = data[f'{frame}_0_recall'][0]
        recall_1 = data[f'{frame}_1_recall'][0]
        f1_0 = data[f'{frame}_0_f1-score'][0]
        f1_1 = data[f'{frame}_1_f1-score'][0]
        supp_0 = data[f'{frame}_0_support'][0]
        supp_1 = data[f'{frame}_1_support'][0]
        recall_micro = data[f'{frame}_weighted avg_recall'][0]
        press_micro = data[f'{frame}_weighted avg_precision'][0]
        f1_micro = data[f'{frame}_weighted avg_f1-score'][0]
        supp_micro = data[f'{frame}_weighted avg_support'][0]
        recall_macro = data[f'{frame}_macro avg_recall'][0]
        press_macro = data[f'{frame}_macro avg_precision'][0]
        f1_macro = data[f'{frame}_macro avg_f1-score'][0]
        supp_macro = data[f'{frame}_macro avg_support'][0]
        acc = data[f'{frame}_acc'][0]

        std_pres_0 = data_std[f'{frame}_0_precision'][0]
        std_pres_1 = data_std[f'{frame}_1_precision'][0]
        std_recall_0 = data_std[f'{frame}_0_recall'][0]
        std_recall_1 = data_std[f'{frame}_1_recall'][0]
        std_f1_0 = data_std[f'{frame}_0_f1-score'][0]
        std_f1_1 = data_std[f'{frame}_1_f1-score'][0]
        std_supp_0 = data_std[f'{frame}_0_support'][0]
        std_supp_1 = data_std[f'{frame}_1_support'][0]
        std_recall_micro = data_std[f'{frame}_weighted avg_recall'][0]
        std_press_micro = data_std[f'{frame}_weighted avg_precision'][0]
        std_f1_micro = data_std[f'{frame}_weighted avg_f1-score'][0]
        std_supp_micro = data_std[f'{frame}_weighted avg_support'][0]
        std_recall_macro = data_std[f'{frame}_macro avg_recall'][0]
        std_press_macro = data_std[f'{frame}_macro avg_precision'][0]
        std_f1_macro = data_std[f'{frame}_macro avg_f1-score'][0]
        std_supp_macro = data_std[f'{frame}_macro avg_support'][0]
        std_acc = data_std[f'{frame}_acc'][0]
        
        auc = data[f'{frame}_auc_press_recall'][0]
        std_auc = data_std[f'{frame}_auc_press_recall'][0]
        roc = data[f'{frame}_roc_auc_score'][0]
        std_roc = data_std[f'{frame}_roc_auc_score'][0]

        all_auc.append(auc)
        all_std_auc.append(std_auc)
        all_roc.append(roc)
        all_std_roc.append(std_roc)
        
        all_acc.append(acc)
        
        all_recall_micro.append(recall_micro)
        all_recall_macro.append(recall_macro)
        all_pre_micro.append(press_micro)
        all_pre_macro.append(press_macro)
        all_f1_micro.append(f1_micro)
        all_f1_macro.append(f1_macro)
        
        all_std_recall_micro.append(std_recall_micro)
        all_std_recall_macro.append(std_recall_macro)
        all_std_pre_micro.append(std_press_micro)
        all_std_pre_macro.append(std_press_macro)
        all_std_f1_micro.append(std_f1_micro)
        all_std_f1_macro.append(std_f1_macro)


    recall_micro = np.mean(all_recall_micro)
    recall_macro = np.mean(all_recall_macro)
    press_micro = np.mean(all_pre_micro)
    press_macro = np.mean(all_pre_macro)
    f1_micro = np.mean(all_f1_micro)
    f1_macro = np.mean(all_f1_macro)
    
    auc = np.mean(all_auc)
    std_auc = np.mean(all_std_auc)
    
    roc = np.mean(all_roc)
    std_roc = np.mean(all_std_roc)
    
    std_recall_micro = np.mean(all_std_recall_micro)
    std_recall_macro = np.mean(all_std_recall_macro)
    std_press_micro = np.mean(all_std_pre_micro)
    std_press_macro = np.mean(all_std_pre_macro)
    std_f1_micro = np.mean(all_std_f1_micro)
    std_f1_macro = np.mean(all_std_f1_macro)

    print("""
        {:20s}
        {:>12s} {:>16s} {:>16s} {:>16s} {:>16s} {:>16s}
        
        {:>12s} {:10.2f}(±{:4.2f}) {:10.2f}(±{:4.2f}) {:10.2f}(±{:4.2f})
        {:>12s} {:10.2f}(±{:4.2f}) {:10.2f}(±{:4.2f}) {:10.2f}(±{:4.2f}) {:10.2f}(±{:4.2f}) {:10.2f}(±{:4.2f})
        """.format("Mean",
                   " ", "precision", "recall", "f1-score", "AUC", "ROC AUC",
                   "Micro", press_micro, std_press_micro, recall_micro, std_recall_micro, f1_micro, std_f1_micro,
                   "Macro", press_macro, std_press_macro, recall_macro, std_recall_macro, f1_macro, std_f1_macro, auc, std_auc, roc, std_roc
                   
        ))
    
    
    
def calculate_pres_recall(preds, Y):
    pres_class = [0, 0, 0, 0]
    recall_class = [0, 0, 0, 0]
    acc_class = [0, 0, 0, 0]

    all_y_pred = []
    all_y_true = []
    for i in range(4):
        y_pred = [int(pred[i]) for pred in preds]
        y_true = [int(target[i]) for target in Y]

        all_y_pred.extend(y_pred)
        all_y_true.extend(y_true)

        pres_class[i] = precision_score(y_true, y_pred, zero_division=0)
        recall_class[i] = recall_score(y_true, y_pred, zero_division=0)
        acc_class[i] = accuracy_score(y_true, y_pred)

    mean_pres = precision_score(all_y_true, all_y_pred, zero_division=0)
    mean_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    mean_acc = accuracy_score(all_y_true, all_y_pred)

    return mean_pres, mean_recall, mean_acc, pres_class, recall_class, acc_class 


def save_data(writer, all_logits, Y, total_loss, loss_class, total_data, fold_index, epoch, step):
    if writer is None:
        return

    loss = (total_loss/total_data).item()
    pres, recall, acc, pres_class, recall_class, acc_class = calculate_pres_recall(all_logits, Y)

    writer.add_scalar(f'Fold_{fold_index}/loss_{step}', loss, epoch)
    writer.add_scalar(f'Fold_{fold_index}/recall_{step}', recall, epoch)
    writer.add_scalar(f'Fold_{fold_index}/presicion_{step}', pres, epoch)
    writer.add_scalar(f'Fold_{fold_index}/acc_{step}', acc, epoch)

    for i in range(len(loss_class)):
        loss_class_train = loss_class[i]/(total_data/4)
        writer.add_scalar(f'Fold_{fold_index}/loss_class_{ORDER[i]}_{step}', loss_class_train, epoch)
        writer.add_scalar(f'Fold_{fold_index}/presicion_{ORDER[i]}_{step}', pres_class[i], epoch)
        writer.add_scalar(f'Fold_{fold_index}/recall_{ORDER[i]}_{step}', recall_class[i], epoch)
        writer.add_scalar(f'Fold_{fold_index}/acc_{ORDER[i]}_{step}', acc_class[i], epoch)

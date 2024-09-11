## Imports ##

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import pickle

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from parameter_tuning import ClassificationModel, get_initial_parameter_list
from parameter_tuning import parameter_tuning_rust, parameter_tuning_drainage, parameter_tuning_graffiti
from parameter_tuning import parameter_tuning_exposed_rebars, parameter_tuning_wetspot, parameter_tuning_crack


## Cross Classification model ##

class CrossClassificationModel():

    def __init__(self, relevant_features):

        self.all_features = ["darker", "gradient", "reddish", "metallic", "colorful", "black", "black_thin", "dominating_color", "color_bin_1", 
                             "color_bin_2", "color_bin_3", "color_bin_4", "color_bin_5", "color_bin_6", "color_bin_7", "color_bin_8", "color_bin_9", 
                             "color_entropy", "rough", "dominating_texture", "texture_0", "texture_1", "texture_2", "texture_3", "texture_4", 
                             "texture_5", "texture_6", "texture_7", "texture_8", "texture_9", "rough_entropy", "lengthy", "number_lengthy_objects", 
                             "lengthy_aspect_ratio", "rel_length", "in_shape", "roundness", "hu_moment_1", "hu_moment_2", "hu_moment_3", 
                             "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"]

        self.initial_parameter_list = get_initial_parameter_list()
        self.relevant_features = relevant_features
        self.classifiers = self.load_classifiers()
        self.parameter_lists = self.load_parameter_lists()
        self.predictions, self.labels = self.predict_binary()

        self.results = pd.concat([self.predictions.reset_index(drop=True), 
                                  self.predict_label().reset_index(drop=True), 
                                  self.labels.reset_index(drop=True)], axis=1)

    def load_classifiers(self):

        classifiers = {}

        for defect in self.relevant_features.keys():
            try:
                with open(f"parameter_tuning/classifiers/resulting_classifier_{defect}.pkl", "rb") as f:
                    clf = pickle.load(f)
            except:
                try:
                    results = pd.read_csv(f"parameter_tuning/results/results_parameter_tuning_{defect}.csv")
                    parameter_list = results.loc[0, "Best dev parameter list"]
                    parameter_list = [int(x) for x in parameter_list.strip("()").split(", ")]
                except:
                    parameter_list = self.initial_parameter_list
                defect_classifier = ClassificationModel(defect, features=self.relevant_features[defect])
                _, _, _, clf = defect_classifier.run(parameter_list)
            classifiers[defect] = clf
        
        return classifiers
    
    def load_parameter_lists(self):

        parameter_lists = {}

        for defect in self.relevant_features.keys():
            try:
                results = pd.read_csv(f"parameter_tuning/results/results_parameter_tuning_{defect}.csv")
                parameter_list = results.loc[0, "Best dev parameter list"]
                parameter_list = [int(x) for x in parameter_list.strip("()").split(", ")]
            except:
                parameter_list = self.initial_parameter_list
            parameter_lists[defect] = parameter_list
        
        return parameter_lists

    def predict_binary(self):

        predictions = {}
        
        for defect in self.relevant_features.keys():
            classifier = ClassificationModel(defect, features=self.relevant_features[defect])
            _, _, _, probs, labels = classifier.test(self.parameter_lists[defect], self.classifiers[defect], balanced_sampling=False)
            predictions[defect] = probs

        predictions = pd.DataFrame(predictions)
        return predictions, labels
    
    def predict_label(self):

        predicted_labels = {}

        for i in range(len(self.predictions)):
            count = 0
            max = 0
            max_label = None
            for j in range(self.predictions.shape[1]):
                if self.predictions.iloc[i, j] >= 0.5:
                    count += 1
                    if self.predictions.iloc[i, j] > max:
                        max = self.predictions.iloc[i, j]
                        max_label = self.predictions.columns[j]
            predicted_labels[i] = [count, max, max_label if not max_label == None else "Nothing"]
        
        predicted_labels = pd.DataFrame(predicted_labels).T
        predicted_labels.columns = ["Count", "Max", "Predicted Label"]

        return predicted_labels
    
    def evaluate_results(self, variant):

        y_true = self.results["label"]
        y_pred = self.results["Predicted Label"]

        for i, _ in enumerate(y_true):
            if y_true.iloc[i] not in self.relevant_features.keys():
                y_true.iloc[i] = "Nothing"

        accuracy = accuracy_score(y_true, y_pred)
        
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')

        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # Save results:
        df = pd.DataFrame({"Accuracy": [accuracy], "Recall (Micro)": [recall_micro], "Recall (Macro)": [recall_macro], 
                           "Recall (Weighted)": [recall_weighted], "F1 Score (Micro)": [f1_micro], "F1 Score (Macro)": [f1_macro], 
                           "F1 Score (Weighted)": [f1_weighted]})
        df.to_csv(f"parameter_tuning/results/results_cross_classification_{variant}.csv")

        cm = confusion_matrix(y_true, y_pred, labels = sorted(list(set(y_true))))
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        # Compute confusion matrix
        labels = sorted(list(set(y_true)))  # Get unique labels and sort them

        # Plotting the confusion matrix as a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        # Save the plot
        os.makedirs(os.path.dirname(f"parameter_tuning/results/results_cross_classification_{variant}.png"), exist_ok=True)
        plt.savefig(f"parameter_tuning/results/results_cross_classification_{variant}.png")
        plt.close()


## Evaluation ##

def cross_classification_evaluation():

    all_features = ["darker", "gradient", "reddish", "metallic", "colorful", "black", "black_thin", "dominating_color", "color_bin_1", 
                    "color_bin_2", "color_bin_3", "color_bin_4", "color_bin_5", "color_bin_6", "color_bin_7", "color_bin_8", "color_bin_9", 
                    "color_entropy", "rough", "dominating_texture", "texture_0", "texture_1", "texture_2", "texture_3", "texture_4", 
                    "texture_5", "texture_6", "texture_7", "texture_8", "texture_9", "rough_entropy", "lengthy", "number_lengthy_objects", 
                    "lengthy_aspect_ratio", "rel_length", "in_shape", "roundness", "hu_moment_1", "hu_moment_2", "hu_moment_3", 
                    "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"]

    relevant_features_focus = {"Rust": parameter_tuning_rust(return_features=True), 
                               "Drainage": parameter_tuning_drainage(return_features=True), 
                               "Graffiti": parameter_tuning_graffiti(return_features=True), 
                               "ExposedRebars": parameter_tuning_exposed_rebars(return_features=True), 
                               "Wetspot": parameter_tuning_wetspot(return_features=True), 
                               "Crack": parameter_tuning_crack(return_features=True)
                               }
    
    relevant_features_all = {"Rust": parameter_tuning_rust(return_features=True), 
                             "Drainage": parameter_tuning_drainage(return_features=True), 
                             "Graffiti": parameter_tuning_graffiti(return_features=True), 
                             "ExposedRebars": parameter_tuning_exposed_rebars(return_features=True), 
                             "Wetspot": parameter_tuning_wetspot(return_features=True), 
                             "Crack": parameter_tuning_crack(return_features=True), 
                             "Weathering": all_features,  
                             "Rockpocket": all_features,  
                             "Spalling": all_features,  
                             "WConccor": all_features,  
                             "Cavity": all_features,  
                             "Efflorescence": all_features,  
                             "PEquipment": all_features,  
                             "Bearing": all_features,  
                             "Hollowareas": all_features,  
                             "JTape": all_features,  
                             "Restformwork": all_features, 
                             "ACrack": all_features, 
                             "EJoint": all_features
                             }

    CrossClassificationModel(relevant_features_focus).evaluate_results("focus")
    CrossClassificationModel(relevant_features_all).evaluate_results("all")


if __name__ == "__main__":

    cross_classification_evaluation()

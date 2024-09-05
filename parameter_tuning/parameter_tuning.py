## Imports ##

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pickle

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from image_processing import Dataset

# Seeds
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


## Binary Classification model ##

class ClassificationModel():

    def __init__(self, target_label, features=None):

        set_seed(42)
        self.all_features = ["darker", "gradient", "reddish", "metallic", "colorful", "black", "black_thin", "dominating_color", "color_bin_1", 
                             "color_bin_2", "color_bin_3", "color_bin_4", "color_bin_5", "color_bin_6", "color_bin_7", "color_bin_8", "color_bin_9", 
                             "color_entropy", "rough", "dominating_texture", "texture_0", "texture_1", "texture_2", "texture_3", "texture_4", 
                             "texture_5", "texture_6", "texture_7", "texture_8", "texture_9", "rough_entropy", "lengthy", "number_lengthy_objects", 
                             "lengthy_aspect_ratio", "rel_length", "in_shape", "roundness", "hu_moment_1", "hu_moment_2", "hu_moment_3", 
                             "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"]

        self.features = features if not features == None else self.all_features
        self.target_label = target_label
        self.measures = {}
        self.classifiers = {}
        self.resolution = 500
        self.dataset = None

    def generate_dataset(self, parameter_list):

        if self.dataset == None:
            dataset = Dataset.new(dataset_type="train", features_list=self.features, parameter_list=parameter_list, resolution=self.resolution)
        else:
            dataset = self.dataset.regenerate(parameter_list)
        self.dataset = dataset
        dataset = dataset.data
        dataset.drop(columns = ["image_number", "defect_number"], inplace = True)
        dataset.dropna(inplace = True)
        x_train, x_test, y_train, y_test = self.create_train_test_data(dataset)

        return x_train, x_test, y_train, y_test
    
    def balanced_sampling(self, target_class_df):

        target_class_df = target_class_df.sort_values(by = "target_label", ascending = False)
        number_of_defects = target_class_df["target_label"].sum()
        target_class_df_1 = target_class_df.iloc[:number_of_defects, :]
        negative_samples = random.sample(range(number_of_defects, len(target_class_df)), number_of_defects)
        target_class_df_0 = target_class_df.iloc[negative_samples, :]
        df = pd.concat([target_class_df_1, target_class_df_0], ignore_index=True)

        return df

    def create_train_test_data(self, dataset):

        # Create target labels:
        target_class_df = dataset.copy()
        target_class_df["target_label"] = target_class_df["label"].apply(lambda x: 1 if x == self.target_label else 0)

        # Apply train test split:
        df = self.balanced_sampling(target_class_df)
        x = df.loc[:, self.all_features]
        y = df.loc[:, "target_label"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        x_train, x_test = self.select_features(self.features, x_train, x_test)

        return x_train, x_test, y_train, y_test
    
    def select_features(self, features, x_train, x_test):

        x_train = x_train.loc[:, features]
        x_test = x_test.loc[:, features]

        return x_train, x_test
    
    def evaluate(self, y_pred, y_test):

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average=None)[1]
        f1 = f1_score(y_test, y_pred, average=None)[1]

        return accuracy, recall, f1
    
    def run(self, parameter_list, number_of_forests=1):

        # Check whether parameter combination already exists in measures dictionary:
        parameter_combination = tuple(parameter_list)
        if parameter_combination in self.measures:
            accuracy, recall, f1 = self.measures[parameter_combination]
            clf = self.classifiers[parameter_combination]
        else:

            # Generate the dataset:
            x_train, x_test, y_train, y_test = self.generate_dataset(parameter_list)

            # Perform the random forest classifier: 
            for i in range(number_of_forests):
                clf = RandomForestClassifier(n_estimators=50, criterion= "entropy", random_state=i, max_depth=10)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                if i == 0:
                    y_pred_final = y_pred
                else:
                    y_pred_final = np.array([1 if y_pred[i] == 1 or y_pred_final[i] == 1 else 0 for i in range(len(y_pred))])

            # Evaluate and store results:
            accuracy, recall, f1 = self.evaluate(y_pred_final, y_test)
            self.measures[parameter_combination] = (accuracy, recall, f1)
            self.classifiers[parameter_combination] = clf

        print(f"Parameter list: {parameter_combination}, \nAccuracy: {accuracy}, \nRecall: {recall}, \nF1: {f1}\n")
        return accuracy, recall, f1, clf

    def test(self, parameter_list, clf):

        # Load and prepare validation dataset
        df = Dataset.new(dataset_type="validation", features_list=self.features, parameter_list=parameter_list, resolution=self.resolution).data
        df.drop(columns = ["image_number", "defect_number"], inplace = True)
        labels = df["label"]
        df.dropna(inplace = True)
        df["target_label"] = df["label"].apply(lambda x: 1 if x == self.target_label else 0)
        x_test, y_test = df.loc[:, self.all_features], df.loc[:, "target_label"]
        x_test, _ = self.select_features(self.features, x_test, x_test)

        # Test and evaluate
        y_pred = clf.predict(x_test)
        y_probas = clf.predict_proba(x_test)[:, 1]
        accuracy, recall, f1 = self.evaluate(y_pred, y_test)

        return accuracy, recall, f1, y_probas, labels


## Environment ##

class State():

    def __init__(self, state, action=None):

        self.value_space = self.get_value_spaces()

        if not action == None:
            state = self.construct(state, action)

        self.parameter_tensor = state

    def get_value_spaces(self):
        
        value_spaces = {
            "0: convex_shape_image_filter: blur_kernel_size": (3, 5, 7, 9, 11, 13, 15),
            "1: convex_shape_image_filter: edge_lower_threshold": (25, 50, 75, 100, 125), 
            "2: convex_shape_image_filter: edge_width": (50, 75, 100, 125, 150),
            "3: convex_shape_image_filter: dilate_kernel_size": (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
            "4: convex_shape_image_filter: erode_kernel_difference": (-4, -3, -2, -1, 0, 1, 2),
            "5: lengthy_image_filter: blur_kernel_size": (3, 5, 7, 9, 11, 13, 15),
            "6: lengthy_image_filter: edge_lower_threshold": (25, 50, 75, 100, 125),
            "7: lengthy_image_filter: edge_width": (50, 75, 100, 125, 150),
            "8: lengthy_image_filter: dilate_kernel_size": (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
            "9: lengthy_image_filter: erode_kernel_difference": (-4, -3, -2, -1, 0, 1, 2),
            "10: darker_image_filter: blur_kernel_size_1": (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
            "11: darker_image_filter: blur_kernel_size_2": (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
            "12: darker_image_filter: z": (100, 120, 140, 160, 180, 200, 220, 240),
            "13: rough_image_filter: blur_kernel_size": (3, 4, 5, 6, 7, 8, 9, 10), 
            "14: rough_image_filter: z_1": (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
            "15: rough_image_filter: binary_blur_kernel_size": (30, 35, 40, 45, 50, 55, 60, 65, 70), 
            "16: rough_image_filter: z_2": (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
            "17: darkness_gradient: scale_outer": (60, 65, 70, 75, 80, 85, 90)
            }
        
        return list(value_spaces.values())

    def construct(self, state, action):

        adjusted_state = state.parameter_tensor.clone()

        for i, action_i in enumerate(action):
            len_value_space = len(self.value_space[i])
            intervals = [(i + 1) / len_value_space for i in range(len_value_space)]
            for j, interval in enumerate(intervals): 
                if action_i <= interval:
                    adjusted_state[i] = self.value_space[i][j]
                    break

        return adjusted_state
    
class Dacl10kEnvironment():

    def __init__(self, classifier, initial_parameter_list=None):
        
        self.classifier = classifier
        self.initial_parameter_tensor = torch.FloatTensor(initial_parameter_list)
        self.initial_state = State(self.initial_parameter_tensor) 
        self.statesize = len(self.initial_state.parameter_tensor)

        self.state = self.initial_state

        self.accuracies = {}
        self.recalls = {}
        self.f1_scores = {}
        self.classifiers = {}

    def step(self, action):

        next_state = State(self.state, action)
        self.state = next_state

        parameter_list = list(self.state.parameter_tensor)
        for i, parameter in enumerate(parameter_list):
            parameter_list[i] = int(parameter)

        accuracy, recall, f1, clf = self.classifier.run(parameter_list)
        reward = f1
        self.accuracies[tuple(parameter_list)] = accuracy
        self.recalls[tuple(parameter_list)] = recall
        self.f1_scores[tuple(parameter_list)] = f1
        self.classifiers[tuple(parameter_list)] = clf

        return next_state, reward
    

## Reinforcement Agent ##

class FeedforwardNet(nn.Module):

    def __init__(self, env, device, hidden_size):

        super(FeedforwardNet, self).__init__()
        
        self.device = device
        self.statesize = env.statesize
        self.hidden_size = hidden_size
        
        self.hidden_layer = nn.Linear(self.statesize, self.hidden_size).to(self.device)
        self.output_layer = nn.Linear(self.hidden_size, self.statesize).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

    def forward(self, input):

        x = self.hidden_layer(input)
        x = torch.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x
    
class ReinforcementAgent():

    def __init__(self, env):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = FeedforwardNet(env, self.device, 30)

        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=0.01)

        self.initial_exploration_chance = 0.5
        self.statesize = env.statesize

    def step(self, state, epoch, epochs):

        action = self.net.forward(state.parameter_tensor)
        #print(action)
        action = action.tolist()
        
        # Exploration adjustements:
        exploration_chance = self.initial_exploration_chance * (((epochs - 1) - epoch) / (epochs - 1))
        for i in range(self.statesize):
            if random.random() < exploration_chance:
                action[i] = random.random()
        
        return action
    
    def update(self, reward):

        reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)
        loss = torch.sqrt(1 - reward)
        #print(f"Loss: {loss}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


## Train the reinforcement agent ##

def plot_rewards(rewards, directory=None):

    x = list(range(len(rewards)))

    moving_average_rewards = []
    window_size = 50
    for i in range(len(rewards)):
        start_index = max(0, i - window_size + 1)
        window = rewards[start_index:i + 1]
        moving_average_rewards.append(sum(window) / len(window))

    plt.figure(figsize=(10, 5))
    plt.plot(x, rewards, marker="o", linestyle="-", color="b")
    plt.plot(x, moving_average_rewards, linestyle="-", color="r")

    plt.title("Rewards")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.grid(True)

    if not directory == None:
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        plt.savefig(directory)
    plt.close()

def train_agent(classifier, initial_parameter_list=None, epochs=20, directory=None):

    # Initialize variables:
    if initial_parameter_list == None:
        initial_parameter_list = [5, 10, 100, 10, -2, 7, 50, 100, 15, 0, 15, 5, 200, 5, 10, 50, 15, 85]
    env = Dacl10kEnvironment(classifier, initial_parameter_list)
    agent = ReinforcementAgent(env)

    state = env.initial_state
    rewards = []

    # Run epochs:
    for i in range(epochs): 

        print(f"Epoch {i}")
        
        action = agent.step(state, i, epochs)
        #print(action)
        next_state, reward = env.step(action)
        agent.update(reward)
        state = next_state

        print(f"Reward: {reward}\n")
        rewards.append(reward)

    # Evaluate results:
    best_parameter_list = max(env.f1_scores, key=lambda k: env.f1_scores[k])
    best_f1_score = env.f1_scores[best_parameter_list]
    accuracy = env.accuracies[best_parameter_list]
    recall = env.recalls[best_parameter_list]
    best_classifier = env.classifiers[best_parameter_list]
    print(f"Best parameter list: {best_parameter_list}\nReward: {best_f1_score}\n")
    plot_rewards(rewards, directory)

    return best_parameter_list, best_f1_score, accuracy, recall, best_classifier


## Experiments ##

def get_initial_parameter_list():
    
    initial_parameter_dict = {
            "0: convex_shape_image_filter: blur_kernel_size": 5,
            "1: convex_shape_image_filter: edge_lower_threshold": 50, 
            "2: convex_shape_image_filter: edge_width": 100,
            "3: convex_shape_image_filter: dilate_kernel_size": 10,
            "4: convex_shape_image_filter: erode_kernel_difference": -2,
            "5: lengthy_image_filter: blur_kernel_size": 7,
            "6: lengthy_image_filter: edge_lower_threshold": 50,
            "7: lengthy_image_filter: edge_width": 100,
            "8: lengthy_image_filter: dilate_kernel_size": 15,
            "9: lengthy_image_filter: erode_kernel_difference": 0,
            "10: darker_image_filter: blur_kernel_size_1": 15,
            "11: darker_image_filter: blur_kernel_size_2": 5,
            "12: darker_image_filter: z": 200,
            "13: rough_image_filter: blur_kernel_size": 5, 
            "14: rough_image_filter: z_1": 10,
            "15: rough_image_filter: binary_blur_kernel_size": 50, 
            "16: rough_image_filter: z_2": 15,
            "17: darkness_gradient: scale_outer": 85
            }
    
    initial_parameter_list = list(initial_parameter_dict.values())

    return initial_parameter_list

def parameter_tuning_experiment(defect_type, defect_features, epochs): 

    initial_parameter_list = get_initial_parameter_list()

    # Perform classification with initial parameter list
    defect_classifier = ClassificationModel(defect_type, features=defect_features)
    _, _, _, initial_clf = defect_classifier.run(initial_parameter_list)
    initial_accuracy, initial_recall, initial_f1, _, _ = defect_classifier.test(initial_parameter_list, initial_clf)

    # Perform reinforcement learning 
    best_dev_parameter_list, best_dev_f1_score, dev_accuracy, dev_recall, best_dev_clf = train_agent(
        defect_classifier, initial_parameter_list=initial_parameter_list, epochs=epochs, 
        directory=f"parameter_tuning/results/results_parameter_tuning_{defect_type}.png")
    test_accuracy, test_recall, test_f1, _, _ = defect_classifier.test(best_dev_parameter_list, best_dev_clf)

    # Save results
    df = pd.DataFrame({"Defect": [defect_type], "Initial f1 score": [initial_f1], "Initial accuracy": [initial_accuracy], 
                       "Initial recall": [initial_recall], "Best dev f1_score": [best_dev_f1_score], "Dev accuracy": [dev_accuracy], 
                       "Dev recall": [dev_recall], "Best dev parameter list": [best_dev_parameter_list], 
                       "Test accuracy": [test_accuracy], "Test recall": [test_recall], 
                       "Test f1": [test_f1]})
    df.to_csv(f"parameter_tuning/results/results_parameter_tuning_{defect_type}.csv")

    # Save best classifier
    os.makedirs(os.path.dirname(f"parameter_tuning/classifiers/resulting_classifier_{defect_type}.pkl"), exist_ok=True)
    with open(f"parameter_tuning/classifiers/resulting_classifier_{defect_type}.pkl", "wb") as f:
        pickle.dump(best_dev_clf, f)


def parameter_tuning_rust(return_features=False):
    
    rust_features = ["darker", "gradient", "reddish", "metallic", "colorful", "black", "black_thin", "dominating_color", "color_bin_1", 
                             "color_bin_2", "color_bin_3", "color_bin_4", "color_bin_5", "color_bin_6", "color_bin_7", "color_bin_8", "color_bin_9", 
                             "color_entropy", "rough", "dominating_texture", "texture_0", "texture_1", "texture_2", "texture_3", "texture_4", 
                             "texture_5", "texture_6", "texture_7", "texture_8", "texture_9", "rough_entropy", "lengthy", "number_lengthy_objects", 
                             "lengthy_aspect_ratio", "rel_length", "in_shape", "roundness", "hu_moment_1", "hu_moment_2", "hu_moment_3", 
                             "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"]
    
    if return_features == True: 
        return rust_features

    parameter_tuning_experiment("Rust", rust_features, epochs=30)

def parameter_tuning_drainage(return_features=False):
    
    drainage_features = ["darker", "gradient", "reddish", "metallic", "colorful", "black", "black_thin", "dominating_color", "color_bin_1", 
                             "color_bin_2", "color_bin_3", "color_bin_4", "color_bin_5", "color_bin_6", "color_bin_7", "color_bin_8", "color_bin_9", 
                             "color_entropy", "rough", "dominating_texture", "texture_0", "texture_1", "texture_2", "texture_3", "texture_4", 
                             "texture_5", "texture_6", "texture_7", "texture_8", "texture_9", "rough_entropy", "lengthy", "number_lengthy_objects", 
                             "lengthy_aspect_ratio", "rel_length", "in_shape", "roundness", "hu_moment_1", "hu_moment_2", "hu_moment_3", 
                             "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"]
    
    if return_features == True: 
        return drainage_features
    
    parameter_tuning_experiment("Drainage", drainage_features, epochs=30)

def parameter_tuning_graffiti(return_features=False):
    
    graffiti_features = ["darker", "gradient", "reddish", "metallic", "colorful", "black", "black_thin", "dominating_color", "color_bin_1", 
                             "color_bin_2", "color_bin_3", "color_bin_4", "color_bin_5", "color_bin_6", "color_bin_7", "color_bin_8", "color_bin_9", 
                             "color_entropy", "rough", "dominating_texture", "texture_0", "texture_1", "texture_2", "texture_3", "texture_4", 
                             "texture_5", "texture_6", "texture_7", "texture_8", "texture_9", "rough_entropy", "lengthy", "number_lengthy_objects", 
                             "lengthy_aspect_ratio", "rel_length", "in_shape", "roundness", "hu_moment_1", "hu_moment_2", "hu_moment_3", 
                             "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"]
    
    if return_features == True: 
        return graffiti_features
    
    parameter_tuning_experiment("Graffiti", graffiti_features, epochs=30)

def parameter_tuning_exposed_rebars(return_features=False):
    
    exposed_rebars_features = ["darker", "gradient", "reddish", "metallic", "colorful", "black", "black_thin", "dominating_color", "color_bin_1", 
                             "color_bin_2", "color_bin_3", "color_bin_4", "color_bin_5", "color_bin_6", "color_bin_7", "color_bin_8", "color_bin_9", 
                             "color_entropy", "rough", "dominating_texture", "texture_0", "texture_1", "texture_2", "texture_3", "texture_4", 
                             "texture_5", "texture_6", "texture_7", "texture_8", "texture_9", "rough_entropy", "lengthy", "number_lengthy_objects", 
                             "lengthy_aspect_ratio", "rel_length", "in_shape", "roundness", "hu_moment_1", "hu_moment_2", "hu_moment_3", 
                             "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"]
    
    if return_features == True: 
        return exposed_rebars_features
    
    parameter_tuning_experiment("ExposedRebars", exposed_rebars_features, epochs=30)

def parameter_tuning_wetspot(return_features=False):
    
    wetspot_features = ["darker", "gradient", "reddish", "metallic", "colorful", "black", "black_thin", "dominating_color", "color_bin_1", 
                             "color_bin_2", "color_bin_3", "color_bin_4", "color_bin_5", "color_bin_6", "color_bin_7", "color_bin_8", "color_bin_9", 
                             "color_entropy", "rough", "dominating_texture", "texture_0", "texture_1", "texture_2", "texture_3", "texture_4", 
                             "texture_5", "texture_6", "texture_7", "texture_8", "texture_9", "rough_entropy", "lengthy", "number_lengthy_objects", 
                             "lengthy_aspect_ratio", "rel_length", "in_shape", "roundness", "hu_moment_1", "hu_moment_2", "hu_moment_3", 
                             "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"]
    
    if return_features == True: 
        return wetspot_features
    
    parameter_tuning_experiment("Wetspot", wetspot_features, epochs=30)

def parameter_tuning_crack(return_features=False):
    
    crack_features = ["darker", "gradient", "reddish", "metallic", "colorful", "black", "black_thin", "dominating_color", "color_bin_1", 
                             "color_bin_2", "color_bin_3", "color_bin_4", "color_bin_5", "color_bin_6", "color_bin_7", "color_bin_8", "color_bin_9", 
                             "color_entropy", "rough", "dominating_texture", "texture_0", "texture_1", "texture_2", "texture_3", "texture_4", 
                             "texture_5", "texture_6", "texture_7", "texture_8", "texture_9", "rough_entropy", "lengthy", "number_lengthy_objects", 
                             "lengthy_aspect_ratio", "rel_length", "in_shape", "roundness", "hu_moment_1", "hu_moment_2", "hu_moment_3", 
                             "hu_moment_4", "hu_moment_5", "hu_moment_6", "hu_moment_7"]
    
    if return_features == True: 
        return crack_features
    
    parameter_tuning_experiment("Crack", crack_features, epochs=30)


if __name__ == "__main__":

    parameter_tuning_rust()
    parameter_tuning_drainage()
    parameter_tuning_graffiti()
    parameter_tuning_exposed_rebars()
    parameter_tuning_wetspot()
    parameter_tuning_crack()

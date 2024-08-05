# sample_analysis.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROCESSED_DATA_FOLDER, MODEL_PATH, PATCH_SIZE

class GranzymeBAnalysis:
    def __init__(self, data, threshold):
        self.data = data
        self.threshold = threshold
        self.target_cell1 = "CD8"
        self.target_cell2 = "CD4"
        self.target_cell3 = "CD56"
        self.target_cell4 = "Tumor cell"
        self.marker = "Granzyme B"
        self.rename_columns()
        self.preprocess_data()

    def rename_columns(self):
        self.data.rename(columns={
            'Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)': 'GranzymeB',
            'Nucleus CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)': 'CD8_Nuc',
            'Cytoplasm CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)': 'CD8_Cyt',
            'Membrane CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)': 'CD8_Mem',
            'Nucleus CD4 (Opal 520) Mean (Normalized Counts, Total Weighting)': 'CD4',
            'Nucleus CD56 (Opal 540) Mean (Normalized Counts, Total Weighting)': 'CD56',
            'Nucleus Nucleus (DAPI) Mean (Normalized Counts, Total Weighting)': 'DAPI',
            'Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)': 'Auto'
        }, inplace=True)

    def preprocess_data(self):
        relevant_columns = [
            'GranzymeB', 'CD8_Nuc', 'CD8_Cyt', 'CD8_Mem', 'CD4', 'CD56', 'DAPI', 'Auto'
        ]
        self.data = self.data[relevant_columns + ['True Label', 'Phenotype']]
        self.data[relevant_columns] = self.data[relevant_columns].apply(pd.to_numeric, errors='coerce')
        self.data.dropna(inplace=True)
        scaler = MinMaxScaler()
        self.data[relevant_columns] = scaler.fit_transform(self.data[relevant_columns]) + 0.000001

    def visualize_granzyme_b_distribution(self):
        sns.histplot(self.data['GranzymeB'], kde=True)
        plt.axvline(self.threshold, color='r', linestyle='--')
        plt.title('Distribution of Granzyme B Mean Intensity')
        plt.xlabel('Granzyme B Mean Intensity')
        plt.ylabel('Frequency')
        plt.show()

    def train_classifier(self):
        X = self.data.drop(columns=['True Label', 'Phenotype'])
        y = self.data['True Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        scores = cross_val_score(model, X, y, cv=5)
        mean_score = scores.mean()
        std_score = scores.std()
        print(f'Model Accuracy: {accuracy}')
        print(f'Mean Cross-Validation Score: {mean_score}')
        print(f'Standard Deviation of Cross-Validation Scores: {std_score}')
        return model, mean_score, std_score

    def classify_and_visualize(self, model):
        self.data['Predicted Label'] = model.predict(self.data.drop(columns=['True Label', 'Phenotype']))
        target_labels = [self.target_cell1, self.target_cell2, self.target_cell3, self.target_cell4]
        for target_label in target_labels:
            target_data = self.data[self.data['Phenotype'] == target_label]
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(target_data['CD8_Nuc'], target_data['GranzymeB'], s=0.1, c=(target_data['GranzymeB'] >= self.threshold).astype(int), cmap='coolwarm')
            plt.xlim(0.00005, 5)
            plt.ylim(0.0000005, 5)
            plt.axhline(y=self.threshold, color='r', linestyle='dashed')
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel('Granzyme B Mean Intensity', fontsize=25)
            plt.xlabel('CD8 Mean Intensity', fontsize=25)
            plt.title(f'For {target_label} cell type', fontsize=30)
            CD8Y = target_data['CD8_Nuc']
            Quadrant_1 = target_data[(CD8Y >= self.threshold)].shape[0] / target_data.shape[0]
            Quadrant_4 = target_data[(CD8Y < self.threshold)].shape[0] / target_data.shape[0]
            plt.text(0.5, 1, '{:.1%}'.format(Quadrant_1), fontsize=25)
            plt.text(0.5, 0.000001, '{:.1%}'.format(Quadrant_4), fontsize=25)
            plt.show()

            fig = plt.figure(figsize=(8, 6))
            plt.scatter(target_data['DAPI'], target_data['GranzymeB'], s=0.1, c=(target_data['GranzymeB'] >= self.threshold).astype(int), cmap='coolwarm')
            plt.xlim(0.00005, 5)
            plt.ylim(0.0000005, 5)
            plt.axhline(y=self.threshold, color='r', linestyle='dashed')
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel('Granzyme B Mean Intensity', fontsize=25)
            plt.xlabel('DAPI Mean Intensity', fontsize=25)
            plt.title(f'For {target_label} cell type', fontsize=30)
            DAPIY = target_data['DAPI']
            Quadrant_1 = target_data[(DAPIY >= self.threshold)].shape[0] / target_data.shape[0]
            Quadrant_4 = target_data[(DAPIY < self.threshold)].shape[0] / target_data.shape[0]
            plt.text(0.5, 1, '{:.1%}'.format(Quadrant_1), fontsize=25)
            plt.text(0.5, 0.000001, '{:.1%}'.format(Quadrant_4), fontsize=25)
            plt.show()

    def run(self):
        self.visualize_granzyme_b_distribution()
        model, mean_score, std_score = self.train_classifier()
        self.classify_and_visualize(model)

# Example usage:
inform_files = '/Rebeca&Laura/inform_in_excel/'
granzyme_b_image_folder = '/UEC, CD8 and GranzymeB/'
processed_data_folder = 'data/processed_data'
model_path = 'new_15epochs_model.pth'
patch_size = 256
example_sample = 'Opal 221_8'

inference = InferenceWithThreshold(inform_files, granzyme_b_image_folder, processed_data_folder, model_path, patch_size)
optimal_threshold, inform_data = inference.run(example_sample)

granzyme_b_analysis = GranzymeBAnalysis(inform_data, optimal_threshold)
granzyme_b_analysis.run()

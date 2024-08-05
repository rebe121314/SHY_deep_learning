# utils/model_utils.py
import os
import pickle

class DataHandler:
    @staticmethod
    def save_processed_data(sample_name, image_name, data, processed_data_folder):
        sample_folder = os.path.join(processed_data_folder, sample_name)
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        file_path = os.path.join(sample_folder, f'{image_name}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_processed_data(sample_name, image_name, processed_data_folder):
        file_path = os.path.join(processed_data_folder, sample_name, f'{image_name}.pkl')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        return None

    @staticmethod
    def global_inform_values(inform_df):
        mean_nuclei = np.mean(inform_df['Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)'])
        std_nuclei = np.std(inform_df['Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)'])
        auto_98 = mean_nuclei + 2 * std_nuclei
        return auto_98

    @staticmethod
    def label_relevant_cells(relevant_rows, auto_98):
        aut_nuclei = 'Nucleus Autofluorescence Mean (Normalized Counts, Total Weighting)'
        cd8_mem = 'Membrane CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        cd8_cyt = 'Cytoplasm CD8 (Opal 570) Mean (Normalized Counts, Total Weighting)'
        gb_mem = 'Membrane Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'
        gb_cyt = 'Cytoplasm Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'
        gb_ent = 'Entire Cell Granzyme B (Opal 650) Mean (Normalized Counts, Total Weighting)'

        relevant_cells = relevant_rows.copy()
        relevant_cells['Label'] = 'None'

        relevant_cells[aut_nuclei] = pd.to_numeric(relevant_cells[aut_nuclei], errors='coerce')
        relevant_cells[cd8_mem] = pd.to_numeric(relevant_cells[cd8_mem], errors='coerce')
        relevant_cells[cd8_cyt] = pd.to_numeric(relevant_cells[cd8_cyt], errors='coerce')
        relevant_cells[gb_mem] = pd.to_numeric(relevant_cells[gb_mem], errors='coerce')
        relevant_cells[gb_cyt] = pd.to_numeric(relevant_cells[gb_cyt], errors='coerce')
        relevant_cells[gb_ent] = pd.to_numeric(relevant_cells[gb_ent], errors='coerce')

        # Add the labels to the relevant cells dataframe
        for index, row in relevant_cells.iterrows():
            if row[gb_ent] > 0 and row['Phenotype'] != 'Tumor cell':
                if row[cd8_mem] > row[cd8_cyt]:
                    if row[gb_cyt] > row[cd8_cyt]:
                        relevant_cells.at[index, 'Label'] = 'gb'
            else:
                relevant_cells.at[index, 'Label'] = 'None'

        positive_gb_cells = relevant_cells[relevant_cells['Label'] == 'gb']
        return positive_gb_cells

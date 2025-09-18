from constants import SGD_GOLD_STANDARD_PATH

class Evaluation:
    def __init__(self, gold_standard_path=SGD_GOLD_STANDARD_PATH, ppi_data_loader=None):
    
        self.reference_complexes = []
        with open(gold_standard_path, 'r') as f:
            for line in f.readlines():
                self.reference_complexes.append(line.split())
        
        self.filtered_complexes = self.reference_complexes
        self.ppi_data_loader = ppi_data_loader
    
    def filter_reference_complex(self, filtering_method='just_keep_dataset_proteins', common_threshold=2):
        filtered_complexes = []
        if filtering_method == 'all_proteins_in_dataset':
            for complex in self.reference_complexes:
                flag = True
                for protein in complex:
                    if protein not in self.ppi_data_loader.proteins:
                        flag=False
                        break
                if flag:
                    filtered_complexes.append(complex)
        
        elif filtering_method == 'keep_common_with_threshold':
            for complex in self.reference_complexes:
                counter = 0
                for protein in complex:
                    if protein in self.ppi_data_loader.proteins:
                        counter += 1
                if counter >= common_threshold:
                    filtered_complexes.append(complex)

        elif filtering_method == 'just_keep_dataset_proteins':
            for complex in self.reference_complexes:
                complex_temp = []
                for protein in complex:
                    if protein in self.ppi_data_loader.proteins:
                        complex_temp.append(protein)
                if len(complex_temp) > 2:
                    filtered_complexes.append(complex_temp)
        else:
            raise 'Invalid Filtering method'
        
        self.filtered_complexes = filtered_complexes
    
    def evalute(self, algorithm_complexes, threshold_na=0.25):
        NCP = 0
        NCB = 0
    
        algorithm_to_reference = {}
        for a_idx, a_comp in enumerate(algorithm_complexes):
            for idx, comp in enumerate(self.filtered_complexes):
                if len(a_comp) == 0:
                    continue
                a_comp = set(a_comp)
                comp = set(comp)
                NA = len(a_comp.intersection(comp))**2 / (len(a_comp) * len(comp))
                if NA >= threshold_na:
                    NCP+=1
                    algorithm_to_reference[a_idx] = idx
                    break
    
        reference_to_algorithm = {}
        for idx, comp in enumerate(self.filtered_complexes):
            for a_idx, a_comp in enumerate(algorithm_complexes):
                if len(a_comp) == 0:
                    continue
                a_comp = set(a_comp)
                comp = set(comp)
                NA = len(a_comp.intersection(comp))**2 / (len(a_comp) * len(comp)) 
                if NA >= threshold_na:
                    NCB+=1
                    reference_to_algorithm[idx] = a_idx
                    break
        
        recall = NCB/len(self.filtered_complexes)
        precision = NCP/len(algorithm_complexes)
        return {
            'NCP':NCP,
            'NCB':NCB,
            'Recal':recall,
            'Precision':precision
        }
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
    
    def _precision_score(self, predicted_complex, reference_complex, predicted_num):
        match_info_list = []
        unmatch_pred_list = []
        number=0
        for i, pred in enumerate(predicted_complex):
            overlapscore=0.0
            tmp_max_score_info = None
            for j, ref in enumerate(reference_complex):
                set1 = set(ref)
                set2 = set(pred)
                overlap = set1 & set2
                score = float((pow(len(overlap), 2))) / float((len(set1) * len(set2)))
                if score > overlapscore:
                    overlapscore = score
                    # find max score
                    tmp_max_score_info = {
                        'pred': pred, 'true': ref, 
                        'overlap_score': overlapscore,
                        'pred_id': i, 'true_id': j, 
                    }
            if overlapscore > 0.25:
                number = number + 1
                if tmp_max_score_info is not None:
                    match_info_list.append(tmp_max_score_info)                
            else:
                unmatch_pred_list.append(pred)

        return number/(predicted_num+1e-4), number, match_info_list, unmatch_pred_list
    
    def _recall_score(self, predicted_complex, reference_complex, reference_num):
        match_info_list = []
        unmatch_pred_list = []
        c_number = 0
        for i, ref in enumerate(reference_complex):
            overlapscore=0.0
            tmp_max_score_info = None
            for j, pred in enumerate(predicted_complex):
                set1 = set(ref)
                set2 = set(pred)
                overlap = set1 & set2
                score = float((pow(len(overlap), 2))) / float((len(set1) * len(set2)))
            
                if score > overlapscore:
                    overlapscore = score

                    tmp_max_score_info = {
                        'pred': pred, 'true': ref, 
                        'overlap_score': overlapscore,
                        'pred_id': j, 'true_id': i, 
                    }
            
            if overlapscore > 0.25:
                c_number = c_number+1
                if tmp_max_score_info is not None:
                    match_info_list.append(tmp_max_score_info)
            else:
                unmatch_pred_list.append(pred)

        return c_number/(reference_num+1e-4), c_number, match_info_list, unmatch_pred_list


    def _acc_score(self, predicted_complex, reference_complex):
        # sn
        T_sum1=0.0
        N_sum=0.0   # the number of proteins in reference complex
        for i in reference_complex:
            max=0.0
            for j in predicted_complex:
                set1=set(i)
                set2=set(j)
                overlap=set1&set2
                if len(overlap)>max:
                    max=len(overlap)
            T_sum1=T_sum1+max
            N_sum=N_sum+len(set1)

        # ppv
        T_sum2=0.0
        T_sum=0.0
        for i in predicted_complex:
            max=0.0
            for j in reference_complex:
                set1=set(i)
                set2=set(j)
                overlap=set1&set2
                T_sum=T_sum+len(overlap)
                if len(overlap)>max:
                    max=len(overlap)
            T_sum2=T_sum2+max
    
        Sn = float(T_sum1) / float(N_sum)
        PPV = float(T_sum2) / float(T_sum+1e-5)
        Acc = pow(float(Sn*PPV), 0.5)
        return Acc

    def evalute(self, algorithm_complexes):
        acc = self._acc_score(algorithm_complexes, self.filtered_complexes)
        reference_num = len(self.filtered_complexes)
        predicted_num = len(algorithm_complexes)

        precision, p_num, _, _ = self._precision_score(algorithm_complexes, self.filtered_complexes, predicted_num)
        recall, r_num, _, _ = self._recall_score(algorithm_complexes, self.filtered_complexes, reference_num)

        f1 = float((2*precision*recall)/(precision+recall+1e-5))


        return {
            'Precision': precision,
            'Recall': recall,
            'Acc': acc,
            'F1': f1,
            'NCP':p_num,
            'NCB': r_num,
        }
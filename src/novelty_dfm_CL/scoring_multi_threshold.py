import numpy as np


class Score_Simple():
    def __init__(self, params):
        '''
        Get top X percentile 
        '''
    def compute(self, scores_old, scores_new):
        scores = scores_old/scores_new
        return scores

    def select(self, scores_old, scores_new, num_select_new):
        scores = self.compute(scores_old, scores_new)
        
        inds_sort = np.argsort(scores) # ascending order         
        inds_novel = inds_sort[-int(num_select_new):]
        
        # get max values 
        return inds_novel


class Score_Scaled():
    def __init__(self, params):
        '''
        Get top X percentile 
        '''
        self.w = params.w
    def compute(self, scores_old, scores_new):
        scores = scores_old*(1-self.w) + (1/scores_new)*self.w
        return scores

    def select(self, scores_old, scores_new, num_select_new):
        scores = self.compute(scores_old, scores_new)

        
        inds_sort = np.argsort(scores) # ascending order         
        inds_novel = inds_sort[-int(num_select_new):]
        
        return inds_novel


class Score_OnlyNew():
    def __init__(self, params):
        '''
        Get bottom X percentile --> invert
        '''
    def compute(self, scores_old, scores_new):
        scores = 1/scores_new
        return scores
    def select(self, scores_old, scores_new, num_select_new):
        scores = self.compute(scores_old, scores_new)
        inds_sort = np.argsort(scores) # ascending order         
        inds_novel = inds_sort[-int(num_select_new):]
        
        return inds_novel



class Score_OnlyOld():
    def __init__(self):
        '''
        Get top X percentile 
        '''
    def compute(self, scores_old, num_select_new):
        inds_sort = np.argsort(scores_old) # ascending order         
        inds_novel = inds_sort[-int(num_select_new):]
        
        return inds_novel
    
    
    
    
class Score_Tug_Simple():
    def __init__(self, params):
        '''
        Get top X and bottom X percentile for new and old respectively
        '''
        self.w_old_i = params.w_old_i
        
    def combine_score_old(self, scores_old_i, scores_old_fixed):
        
        scores_old = self.w_old_i*scores_old_i + (1-self.w_old_i)*scores_old_fixed
        
        return scores_old
        
    def compute(self, scores_old, scores_new):
        scores = scores_old/scores_new
        return scores

    def select(self, i, scores_old, scores_new, num_select_new, num_select_old):
        
        if i==0:
            assert scores_new==None
            scores = scores_old            
        else:
            scores = self.compute(scores_old, scores_new)
            
            
        inds_sort = np.argsort(scores) # ascending order 
        
        # print(num_select_new, num_select_old, inds_sort)
        
        inds_novel = inds_sort[-int(num_select_new):]
        
        inds_old = inds_sort[:int(num_select_old)]

        return inds_novel, inds_old
    


# class Score_Simple():
#     def __init__(self, params):
#         '''
#         Get top X percentile 
#         '''
#     def compute(self, scores_old, scores_new):
#         scores = scores_old/scores_new
#         return scores

#     def select(self, scores_old, scores_new, th_percentile_score):
#         scores = self.compute(scores_old, scores_new)
#         threshold=np.percentile(scores, th_percentile_score)
#         # get max values 
#         inds_novel = np.where(scores>threshold)[0] # binary classification (above th is considered to be Novelty (Positive class))
#         return inds_novel


# class Score_Scaled():
#     def __init__(self, params):
#         '''
#         Get top X percentile 
#         '''
#         self.w = params.w
#     def compute(self, scores_old, scores_new):
#         scores = scores_old*(1-self.w) + (1/scores_new)*self.w
#         return scores

#     def select(self, scores_old, scores_new, th_percentile_score):
#         scores = self.compute(scores_old, scores_new)
#         threshold=np.percentile(scores, th_percentile_score)
#         # get max values 
#         inds_novel = np.where(scores>threshold)[0] # binary classification (above th is considered to be Novelty (Positive class))
#         return inds_novel


# class Score_OnlyNew():
#     def __init__(self, params):
#         '''
#         Get bottom X percentile --> invert
#         '''
#     def compute(self, scores_old, scores_new):
#         scores = 1/scores_new
#         return scores
#     def select(self, scores_old, scores_new, th_percentile_score):
#         scores = self.compute(scores_old, scores_new)
#         threshold=np.percentile(scores, th_percentile_score) # get other side 
#         # get max values 
#         inds_novel = np.where(scores>threshold)[0] # binary classification (above th is considered to be Novelty (Positive class))
#         return inds_novel



# class Score_OnlyOld():
#     def __init__(self):
#         '''
#         Get top X percentile 
#         '''
#     def compute(self, scores_old, th_percentile_score):
#         threshold=np.percentile(scores_old, th_percentile_score) # get other side 
#         # get max values 
#         inds_novel = np.where(scores_old>threshold)[0] # binary classification (above th is considered to be Novelty (Positive class))
#         return inds_novel
    
    
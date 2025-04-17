import os
import argparse
import pandas as pd
import numpy as np
import snf
import sklearn.preprocessing as sp
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics.pairwise import cosine_similarity

def Parameter_arguments():
    Parameter = argparse.ArgumentParser()
    Parameter.add_argument('--moicsdatas', nargs='+', help="omics data")
    Parameter.add_argument('--label', help="sample label")
    return Parameter.parse_args()


def calculate_cosine_similarity(data):
        similarity = cosine_similarity(data.values.T)
        similarity = np.nan_to_num(similarity,copy = False)
        similarity_df = pd.DataFrame(similarity,index = data.columns.values, columns = data.columns.values)
        return similarity_df


def diffusion_enhancement(similarity_matrix, feature_matrix):
    """
    Enabling diffusion-enhanced feature fusion
    Parameters:
        similarity_matrix: sample similarity matrix
        feature_matrix:  matrix of features to be enhanced
    Return:
        Diffusion-enhanced feature matrix
    """
    S = similarity_matrix.div(similarity_matrix.sum(axis=1), axis=0)
    S = np.nan_to_num(S, copy=False) 
    
    enhanced_features = pd.DataFrame(np.matmul(S.values, feature_matrix.values),
                                   index=feature_matrix.index.values,
                                   columns=feature_matrix.columns.values)
    
    return enhanced_features

if __name__ == '__main__':
    ###Model Inputs###
    args = Parameter_arguments()
    
    ###Setting the output directory###
    output = os.getcwd() + '/integration_data'
    os.makedirs(output)
    
    ###Data standardisation###
    omics_num = len(args.moicsdatas)
    omics_list = []
    
    for i in range(omics_num):
        data = pd.read_csv(args.moicsdatas[i], header = 0, index_col = 0)
        data = pd.DataFrame(data, index = data.index.values, columns = data.columns.values)
        
        normalization_method = sp.MinMaxScaler()
        omics_data = pd.DataFrame(normalization_method.fit_transform(data.values), index = data.index.values,columns = data.columns.values)
        
        omics_list.append(omics_data)
        
    ###Consolidation of data###
    omics = pd.concat(omics_list, axis = 1)
    label = pd.read_csv(args.label, header = 0, index_col = 0)
    
    ###Sample similarity network fusion###
    print("Start fusing the sample matrix with the sample similarity network")
    sn  = snf.make_affinity(omics_list, metric = 'euclidean', K = 5, mu = 1)#mu越大，网络越平滑
    fn = snf.snf(sn, K = 5)
    fn = pd.DataFrame(fn, index = omics_list[0].index.values, columns = omics_list[0].index.values)
    fn.to_csv("./integration_data/Sample similarity network matirx.csv")
    
    ##Feature Selection###
    print("Start feature select")
    feature_list = []
    for i in range(omics_num):
        selector = SelectPercentile(chi2, percentile=5)
        selector.fit(omics_list[i].values, label.iloc[:,0].values.reshape(-1,1))
        feature_index = [i for i,x in enumerate(selector.get_support()) if x]
        feature = omics_list[i].iloc[:,feature_index]
        feature_list.append(feature)
        feature.to_csv("./integration_data/%s feature select matirx.csv"%i)
    reduced_data = pd.concat(feature_list, axis = 1)
    reduced_data.to_csv("./integration_data/feature select matirx.csv")
    
    ###Fusing the sample matrix with the sample similarity network合###
    print("Applying diffusion enhancement to fuse features with sample network")
    enhanced_data = diffusion_enhancement(fn, reduced_data)
    enhanced_data.to_csv("./integration_data/diffusion_enhanced_features.csv")

    fn_omics = pd.DataFrame(np.matmul(fn.values, reduced_data.values), 
                            index=reduced_data.index.values, 
                            columns=reduced_data.columns.values)
    fn_omics.to_csv("./integration_data/Diffusion_enhanced_feature_matrix.csv")
    
    print("Diffusion enhancement completed")
    



    
    
    

    
        
        
        
    
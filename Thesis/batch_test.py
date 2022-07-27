#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""The batchCosineSimilarity() function is a function for generating cosine similarity values on a dataset in smaller batches 
    (or chunks) of data, and it is intended to be converted to .py file (batch_test.py), incorporated in the file 
    "Cosine_Similarity_Analysis_Function.ipynb", and performed on subsets of the USDA Global Food Products Database (GBFPD). 
    
   The function takes 4 arguments:
   
   vectors: an numpy array or list representing a set of vectors (in the thesis, this was word frequency vector set of 
   ingredient lists)
   
   index_list: A list of indices for referencing ingredient lists. In this thesis, these indices were FoodData Central IDs 
   (FDC IDs)
   
   subset: A string representing a subset of the dataset (should be a value within the "branded_food_category" column
   of the USDA Global Food Products Database (GBFPD) or "Random" for a random sample of the GBFPD)
   
   batches: The number of batches (i.e. chunks) to divide the dataset into. 
   This is also equal to the number of separate cosine similarity matrices created and consilidated into one.
   
   
"""


#Packages used
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from itertools import combinations,chain,product,permutations
import sys
import random

def batchCosineSimilarity(vectors,index_list,subset,batches=1):
    
    arrays = vectors
    
    #Convert list of indices to array
    indices = np.array(index_list)
    
    #Convert list of vectors to array of vectors, split into a number of batches ("batches")
    arrays = np.array_split(arrays,batches)
    indices = np.array_split(indices,batches)
    
    #Perform cosine similarity in batches
    for x in range(batches):
        
        #Split array of vectors & and array of indices into 20 further subsets (this number can be changed as desired) 
        array_subset = arrays[x]
        array_subset = np.array_split(array_subset,20)
        index_subset = indices[x]
        index_subset = np.array_split(index_subset,20)
        
        #Create data frame for storing cosine similarity data for this batch
        cos_sim_df = pd.DataFrame(columns = ["fdc_id_1","fdc_id_2","Cosine_similarity"])
        
        #List of pairs in batch (used in following nested for loop to check if a given combination of vectors has been used)
        array_pairs = []
        
        #Perform cosine similarity on the further subsetted array subset  
        for i in range(len(array_subset)):
            for j in range(len(array_subset)):
                
                #Included to help prevent duplicates as possible
                if i == j or (j,i) in array_pairs:
                    continue
                else:
                    #Combine indices associated with current subset of the array subset and convert to list
                    curr_indices = np.concatenate((index_subset[i],index_subset[j]), axis=0)
                    curr_indices = curr_indices.tolist()
                    
                    #Generate list of index pairs
                    index_pairs = list(itertools.product(curr_indices,repeat=2))
                    
                    #Dataset for storing pairs for the current iteration
                    curr_df = pd.DataFrame(data = index_pairs, columns = ["fdc_id_1","fdc_id_2"])
                    
                    #Perform cosine similarity for the current iteration and line up with index pairs
                    batch = np.concatenate((array_subset[i],array_subset[j]),axis=0)
                    cos_sim = cosine_similarity(batch)
                    cos_sim_flat = list(cos_sim)
                    cos_sim_flat = list(chain(*cos_sim_flat))
                    curr_df["Cosine_similarity"] = cos_sim_flat
                    
                    #Append current iteration dataset to batch datset
                    cos_sim_df = cos_sim_df.append(curr_df,ignore_index=True)
                    array_pairs.append((i,j))
        
        #Drop duplicates from batch dataset
        cos_sim_df = cos_sim_df.drop_duplicates()
        
        #Save batch dataset to csv
        cos_sim_df.to_csv("".join((subset,"_Batch_",str(x),"_CosSim.csv")), sep = ",")
    
    #Create dataframe for storing all cosine similarity data
    CS_dataframes = pd.DataFrame(columns = ["fdc_id_1","fdc_id_2","Cosine_similarity"])
    
    #Load all batches of cosine similarity data and consolidate batches into one
    for i in range(batches):
        curr_df = pd.read_csv("".join((subset,"_Batch_",str(i),"_CosSim.csv")))

        CS_dataframes = CS_dataframes.append(curr_df)
        
        
    #Prepare cosine similarity dataset for saving to CSV 
    dataset = CS_dataframes
    dataset = dataset.drop(columns='Unnamed: 0')
    dataset["fdc_id_1"] = dataset["fdc_id_1"].astype("str")
    dataset["fdc_id_2"] = dataset["fdc_id_2"].astype("str")
    
    #Save Dataset as CSV (can choose to save to file, return dataset as value, or both, as desired)
    dataset.to_csv("".join((subset,"_CosSim_total.csv")))
    
    #Uncomment following file if desiring to return dataset as a value
    #return dataset



def main():
    vectors = np.load("Random_Vectors_Batch_3_test.npy")
    indices = np.arange(0,len(vectors))
    batchCosineSimilarity(vectors,indices,"Random",3)

if __name__ == "__main__":
    main()


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Packages Used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import readability
import time
import numbers
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import itertools
from itertools import combinations,chain,product,permutations
from batch_test import batchCosineSimilarity
import sys

""" The following function performs a cosine similarity analysis of a subset 'subset' of the FDC Global Branded Food Products 
    Database (GBFPD), assuming it's saved to path as 'branded_food.csv'. The cosine_similarity_analysis function takes 
    three arguments (all string arguments):
    
    filename- The path of the file to be opened
    
    subset- The branded food category to be queried in step 2.
    
    label_color- The color used for the points in the graph. This can be set to a hex code (e.g., #OOOFFF) or a named color 
    (Visit https://matplotlib.org/stable/gallery/color/named_colors.html for a list of named colors)
"""

def cosine_similarity_analysis(filename, subset, label_color="blue"):
    """
    Arguments:
    
    filename: The path of the dataset file. This function assumes that this filename corresponds to the FDC branded food dataset
    (branded_food.csv by default)
    
    subset- The subset of the. Can be string, to indicate a section of the dataset where branded_food_category == 'subset', 
    or numeric, to denote a random sample of size 'subset'.
    
    label_color: the color used for the points on the scatter plots. This is set to blue by default, but can be any color with a 
    hex code, or a named color (Visit https://matplotlib.org/stable/gallery/color/named_colors.html for a list of named colors)
"""
    
    
    #Step 1: Import Food Data Central Food Products Database 
    FoodDataCentral = pd.read_csv(filename, low_memory = False)
   
    #Generate list of branded food categories (Optional, uncomment following three lines to view possible branded food categories for analysis)
    #grouped_counts = pd.DataFrame(FoodDataCentral.groupby(['branded_food_category'])['branded_food_category'].count())
    #grouped_counts.columns = ["bfc_count"]
    #grouped_counts.sort_values(by = "bfc_count", ascending = False)[50:70]

    #Step 2: Get Database subset by branded food category and remove rows with empty values or only non-words 
    if isinstance(subset, numbers.Number):
        delim_subset = "Random"
        FoodDataCentral = FoodDataCentral.sample(n=subset, random_state=11)
    else:
        FDC_query = "".join(("branded_food_category == '",subset,"'")) #Note: subset must be in the domain of 'branded_food_category', or else a ValueError is raised.
        delim_subset = subset.replace(" ","-")
        FoodDataCentral = FoodDataCentral.query(FDC_query)
    
    
    FoodDataCentral = FoodDataCentral.dropna(subset = ['ingredients'])
    FoodDataCentral = FoodDataCentral[FoodDataCentral["ingredients"] != "---"]
    FoodDataCentral = FoodDataCentral[FoodDataCentral["ingredients"] != ","]
    
    

    #Step 3: Calculate and get Readability Scores for FoodData Central ingredient lists
    readability_scores = []
    for index, row in FoodDataCentral.iterrows():
        
        #Get ingredient list and convert to lowercase string (required to calculate syllables in readability.getmeasures())
        ingredients = row["ingredients"]
        ingredients = ingredients.lower()
        
        # Set ingredient lists with incalculable readability to NA
        if pd.isna(ingredients) or ingredients in ["---",","]:
            curr_record = (row['fdc_id'], row['gtin_upc'], pd.NA,pd.NA)
            readability_scores.append(curr_record)

        else:
            """Readability.getmeasures() automatically tokenizes the input using spaces as a delimiter for words and /n as a delimiter for sentences. 
                The function then calculates and returns a set of readability measures. In this case,
                we are getting specific measures from the set (Flesch Reading Ease and Dale-Chall Readability) """
            
            #Prepare data for readability measurement
            token_ing = word_tokenize(ingredients)
            ingredients = token_ing
            
            #Get readability measures
            measures = readability.getmeasures(ingredients)
            curr_record = (row['fdc_id'], row['gtin_upc'], row['branded_food_category'],row['ingredients'],
                           measures['readability grades']['Kincaid'],
                           measures['readability grades']['FleschReadingEase'],
                           measures['readability grades']['DaleChallIndex'],
                           measures['sentence info']['words'],
                           measures['sentence info']['complex_words_dc'])
            readability_scores.append(curr_record)

    #Gather data in Pandas DataFrame
    readScores_FDC = pd.DataFrame(data = readability_scores, columns = ["fdc_id", "gtin_upc","branded_food_category","ingredients",
                                                                        "Kincaid_Score","FleschReadingEase","DaleChallIndex",
                                                                        "num_words","complex_words_dc"])
    #Remove NAs and save dataset to CSV
    readScores_FDC = readScores_FDC.dropna()
    readScores_FDC.to_csv("".join((delim_subset,"_FoodData_Central_Readability.csv")), sep=",")
    FoodDataCentral = readScores_FDC


    #Step 4: Generate Matrix of pairwise differences for Flesch Reading Ease, as a 1-d list
    difference_matrix_fl = [[abs(y - x) for x in FoodDataCentral["FleschReadingEase"]] for y in FoodDataCentral["FleschReadingEase"]]
    difference_matrix_fl = list(chain(*difference_matrix_fl))

    #Step 5: Generate Matrix of pairwise differences for Dale Chall Readability, as a 1-d list
    difference_matrix_dc = [[abs(y - x) for x in FoodDataCentral["DaleChallIndex"]] for y in FoodDataCentral["DaleChallIndex"]]
    difference_matrix_dc = list(chain(*difference_matrix_dc))

    #Step 6: Convert FDC IDs from float to string
    FoodDataCentral["fdc_id"] = FoodDataCentral["fdc_id"].astype("str")

    #Get FDC indices for referencing matrices (Optional, useful for capturing the domain (range) of IDs)
    #fdc_indices = dict(enumerate(FoodDataCentral["fdc_id"]))
    #print(fdc_indices)

    #Step 7: Create list of FDC ID pairs (w/repeats)
    fdcID_pairs = list(itertools.product(FoodDataCentral["fdc_id"],repeat=2))
    

    #Step 8: Create and fit word count vectorizer model to ingredient lists 
    documents = list(FoodDataCentral['ingredients'].values)  #Gather list of ingredient lists from dataset
    count_vectorizer = CountVectorizer(documents, stop_words='english') #Create Count Vectorizer Model
    count_vectorizer.fit(documents)  #Fit model to ingredient list

    
    #Step 9: Transform model to array
    documents_1 = list(FoodDataCentral['ingredients'].values) 
    vectors = count_vectorizer.transform(documents_1).toarray()
    np.save("".join((delim_subset,"_IngredientList_Vectors.npy")),vectors)
    
    

    #Step 10: Calculate cosine similarity of ingredient lists
    cos_sim = cosine_similarity(vectors)
    cos_sim_flat = list(cos_sim)
    cos_sim_flat = list(chain(*cos_sim_flat))
    
    

    #Step 11: Create Dataframe with readability differences and cosine similarities
    analysis_set = pd.DataFrame(fdcID_pairs, columns = ["fdc_id_1","fdc_id_2"])
    analysis_set["DaleChallDiff"] = difference_matrix_dc
    analysis_set["Cosine_similarity"] = cos_sim_flat
    analysis_set["Flesch_diff"] = difference_matrix_fl
    analysis_set["Subset"] = subset
    
    #Step 10-11 (Alternate): In case of low available memory to allocate, the calculations can be done on smaller chunks of the dataset, then combined:
    
    #Create Dataframe for containing all values of Dale Chall difference, Flesch Reading Ease difference, and cosine similarity
    
    analysis_set = pd.DataFrame(fdcID_pairs, columns = ["fdc_id_1","fdc_id_2"])
    print(analysis_set["fdc_id_1"].head(5))
    analysis_set["DaleChallDiff"] = difference_matrix_dc
    analysis_set["Flesch_diff"] = difference_matrix_fl
    analysis_set["Cosine_similarity"] = np.nan #Needed as a placeholder for coalescing after
    
    indices = list(FoodDataCentral["fdc_id"])
    
    #Perform Cosine similarity in batches
    vectors_df = batchCosineSimilarity(vectors,indices,delim_subset,3) 
    vectors_df = pd.read_csv("".join((delim_subset,"_CosSim_total.csv")), chunksize=100000)
    
    chunks = 1
    for df in vectors_df:
        print("Chunk " + str(chunks))
        chunks += 1
        df = df.drop(columns='Unnamed: 0')
        
        df["fdc_id_1"] = df["fdc_id_1"].astype("str")
        df["fdc_id_2"] = df["fdc_id_2"].astype("str")
        analysis_set = analysis_set.merge(df, on=["fdc_id_1","fdc_id_2"], how="left",suffixes = ('', '_x'))
        analysis_set["Cosine_similarity"] = analysis_set["Cosine_similarity"].combine_first(analysis_set["Cosine_similarity_x"])
        analysis_set = analysis_set.drop(columns="Cosine_similarity_x")
        analysis_set = analysis_set.drop_duplicates()    
        
    
    
    #Step 12: Prepare dataframe for plot generation and Summary Statistics
    analysis_set.dropna(inplace=True)
    analysis_set.replace([np.inf, -np.inf], np.nan, inplace=True)
    analysis_set.dropna(inplace=True)
    analysis_set.to_csv("".join((delim_subset,"_Readability_and_CosineSimilarity_scores.csv")))

    #Step 13: Get Scatter Plot- Dale Chall diff vs Cosine Similarity
    plt.scatter(analysis_set["DaleChallDiff"], analysis_set["Cosine_similarity"], c = label_color)
    plt.xlabel("Dale-Chall Index Pairwise difference")
    plt.ylabel("Cosine Similarity")
    plt.title("Difference In Dale-Chall Index vs Cosine Similarity (" + delim_subset + ")")
    plt.savefig("".join((delim_subset,"_DaleChall_vs_CosineSimilarity.png")))
    plt.show()
    

    #Step 14: Get Scatter Plot- Dale Chall diff vs Flesch diff
    plt.scatter(analysis_set["DaleChallDiff"], analysis_set["Flesch_diff"], c = label_color)
    plt.xlabel("Dale-Chall Pairwise difference")
    plt.ylabel("Flesch Pairwise Difference")
    plt.title("Difference In Dale-Chall Index vs Difference in Flesch Reading Ease (" + delim_subset + ")")
    plt.savefig("".join((delim_subset,"_DaleChall_vs_Flesch.png")))
    plt.show()
    

    #Step 15: Get Scatter Plot- Flesch diff vs Cosine Similarity
    plt.scatter(analysis_set["Flesch_diff"], analysis_set["Cosine_similarity"], c = label_color)
    plt.xlabel("Flesch Pairwise difference")
    plt.ylabel("Cosine Similarity")
    plt.title("Difference In Flesch Reading Ease vs Cosine Similarity (" + delim_subset + ")")
    plt.savefig("".join((delim_subset,"_Flesch_vs_CosineSimilarity.png")))
    plt.show()
    

    #Step 16: Get Pearson Correlations (codes: DC = Dale-Chall, FL = Flesch Reading Ease, CS = Cosine similarity)
    DC_CS_pearson = pearsonr(analysis_set["DaleChallDiff"], analysis_set["Cosine_similarity"])
    FL_CS_pearson = pearsonr(analysis_set["Flesch_diff"], analysis_set["Cosine_similarity"])
    DC_FL_pearson = pearsonr(analysis_set["DaleChallDiff"], analysis_set["Flesch_diff"])
    Pearson_corrs = [DC_CS_pearson, FL_CS_pearson, DC_FL_pearson]
    
    return analysis_set
    

def main():
    #cosine_similarity_analysis("branded_food.csv", 6000, "green")
    #cosine_similarity_analysis("branded_food.csv", "Rice", "blue")
    #cosine_similarity_analysis("branded_food.csv", "Pasta Dinners", "red")
    #cosine_similarity_analysis("branded_food.csv", "Cream", "yellow")
    

if __name__ == "__main__":
    main()


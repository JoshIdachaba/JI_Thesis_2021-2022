# JI_Thesis_2021-2022- Cosine Similarity & Readability

This Repository contains the files used in Joshua Idachaba's Master's thesis. This project involves the analysis of the USDA Global Branded Food Products database (GBFPD) in terms of cosine similarity and readability. This program assumes that you have downloaded the GBFPD as a CSV (**branded_food.csv**) from [FoodData Central.](https://fdc.nal.usda.gov/download-datasets.html)

## Database
The [USDA Global Branded Food Product Database (GBFPD)](https://data.nal.usda.gov/dataset/usda-branded-food-products-database) is a USDA database containing information for branded food products, as part of FoodData Central. The October 2021 version of the dataset was downloaded as a .csv file for this thesis, and this version and other versions of the GBFPD can be downloaded from [FoodData Central](https://fdc.nal.usda.gov/download-datasets.html) under section "Branded Food".

## Folders
### Test
The ["Test"](https://github.com/JoshIdachaba/JI_Thesis_2021-2022/tree/main/Test) folder contains sample code for the cosine similarity analysis, as well as memory usage & runtime data. This folder contains two Jupyter IPython Notebook (.ipynb) files:
* [batch_test](https://github.com/JoshIdachaba/JI_Thesis_2021-2022/blob/main/Test/batch_test.ipynb)- contains code for calculating cosine similarity on the dataset in multiple smaller chunks, if desired. 
* [Cosine_similarity_Analysis_Process](https://github.com/JoshIdachaba/JI_Thesis_2021-2022/blob/main/Test/Cosine_Similarity_Analysis_Process.ipynb)- which outlines the step-by-step process by which the cosine similarity analysis is run. The steps are split into individual cells, 1 per step. For this file, cosine similarity analysis is run on the "Rice" subset of the GBFPD as an example.

### Thesis
The ["Thesis"](https://github.com/JoshIdachaba/JI_Thesis_2021-2022/tree/main/Thesis) folder contains all files necessary to run the cosine similarity analysis script, as well as data & figures for the analysis run on July 24, 2022. The October 2021 version of the GBFPD was downloaded for use in the thesis. This folder contains four Jupyter IPython Notebook (.ipynb) files:
* [Cosine_similarity_Analysis_Process](https://github.com/JoshIdachaba/JI_Thesis_2021-2022/blob/main/Thesis/Cosine_Similarity_Analysis_Process.ipynb)- which outlines the step-by-step process by which the cosine similarity analysis is run. The steps are split into individual cells, 1 per step. 
* [Cosine_similarity_Analysis_Function](https://github.com/JoshIdachaba/JI_Thesis_2021-2022/blob/main/Thesis/Cosine_Similarity_Analysis_Function.ipynb)- which contains the files necessary to define and apply the cosine similarity analysis process. This .ipynb file was converted into a .py file for use in another .ipynb file.
* [batch_test](https://github.com/JoshIdachaba/JI_Thesis_2021-2022/blob/main/Thesis/batch_test.ipynb)- contains code for calculating cosine similarity on the dataset in multiple smaller chunks, if desired.
* [summary_statistics](https://github.com/JoshIdachaba/JI_Thesis_2021-2022/blob/main/Thesis/summary_statistics.ipynb)- Which contains code for gathering summary statistics and Pearson correlations.

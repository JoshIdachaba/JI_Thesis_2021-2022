# JI_Thesis_2021-2022- Cosine Similarity & Readability

This Repository contains the files used in Joshua Idachaba's Master's thes. This project involves the analysis of the USDA Global Branded Food Products database (GBFPD) in terms of cosine similarity and readability. This program assumes that you have downloaded the GBFPD from [](https://fdc.nal.usda.gov/download-datasets.html)

## Database
The [USDA Global Branded Food Product Database (GBFPD)](https://data.nal.usda.gov/dataset/usda-branded-food-products-database) is a USDA database containing information for branded food products. This database **branded_food** dataset is a subset of the Food Product Database containing product information for branded foods sold in the US.

## Folders
### Test
The test folder contains sample code for the cosine similarity analysis, as well as memory usage & runtime data. This folder contains two Jupyter IPython Notebook (.ipynb) files:
* batch_test- contains code for calculating cosine similarity on the dataset in multiple smaller chunks, if desired. 
* Cosine_similarity_Analysis_process- which outlines the step-by-step process by which the cosine similarity analysis is run. The steps are split into individual cells, 1 per step. For this file, cosine similarity analysis is run on the "Rice" subset of the GBFPD as an example.

### Thesis
The ["Thesis"] folder contains all files necessary to run the cosine similarity analysis script, as well as data & figures for the analysis run on July 24, 2022. The October 2021 version of the dataset was downloaded for this This folder contains four Jupyter IPython Notebook (.ipynb) files:
* Cosine_similarity_Analysis_process- which outlines the step-by-step process by which the cosine similarity analysis is run. The steps are split into individual cells, 1 per step. 
* Cosine_similarity_Analysis_function- which contains the files necessary to define and apply the cosine similarity analysis process. This .ipynb file was converted into a .py file for use in another .ipynb file.
* batch_test- contains code for calculating cosine similarity on the dataset in multiple smaller chunks, if desired.
* summary_statistics- Which contains code for gathering summary statistics, as well as Pearson correlations.

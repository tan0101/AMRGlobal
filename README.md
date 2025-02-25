# AMR Global

"Title" by Alexandre Maciel-Guerra, Michelle Baker­, Ruoqi Wang­, Luo Chengchang, Yan Xu, Enzo Guerrero-Araya, Weihua Meng, Ge Wu, Komkiew Pinpimai, Tania Dottorini
 
Any questions should be made to the corresponding author Dr Tania Dottorini (Tania.Dottorini@nottingham.ac.uk)

Thirteen scripts are available:

1. Population Correction: get_weights.py and weighted_chi_square_AMR.py
2. Machine Learning: ML_AMR_pipeline.py and ML_AMR_pipeline_Population_correction.py
3. Feature rate: get_antibiotic_class_RSI.py , get_feature_rate.py and get_feature_rate_Population_correction.py
4. Indicator normalization: get_normalized_indicators.py
5. Feature and Indicator Associations: pearson_correlation_and_linear_regression_analysis.py
6. Indicator Monte Carlo Simulations: get_indicator_forecasting_parameters.py and indicator_forecasting_montecarlo.py
7. Feature Monte Carlo Simulations: feature_forecasting_montecarlo.py
8. Trend Analysis: trend_forecast_analysis.py

# System Requirements

## Software requirements

The project was developed using the Conda v23.1.0 environment.

### OS Requirements

This package is supported for Windows. The package has been tested on the following system: 

* Windows: Windows 11 Pro version 23H2 OS build 22631.3296 Windows Feature Experience Pack 1000.22687.1000.0 


### Python Dependencies

```
python v3.9.15
numpy v1.21.5
pandas v1.4.4
scikit-learn v1.2.1
scipy v1.15.2
networkx v2.8.4
matplotlib v3.6.2
imblearn v0.13.0
biopython v1.81
ete3 v3.1.2
```

# Installation Guide:

## Install from Github
```
git clone https://github.com/tan0101/AMRGlobal
cd VibrioCARE
python setup.py install
```

This takes 1-2 min to build

# Instructions for use

After installing the project, run each available code using _python code_name.py_. For the codes get_weights.py , weighted_chi_square_AMR.py , ML_AMR_pipeline.py , ML_AMR_pipeline_Population_correction.py , get_antibiotic_class_RSI.py , get_feature_rate.py and get_feature_rate_Population_correction.py include after the code name the name of the specie between quotation marks. The other codes will automatically import the corresponding data from the **Data** folder and will produce the following output

* Lineage_Separation.py: produces an excel xlsx file named "Features_lineages.xlsx" with the Fisher Exact test statistics for the accessory genes, and core and intergenic SNPs. It will also print in the terminal the Mann Whitney U tests comparing the count of accessory genes, core and intergenic SNPs between the lineages BD-1.2 and BD-2 and the different collection years. This takes 1 min to run
* snp_network.py: produces an SVG figure name "SNP_network_Vibrio.svg" (Figure 2 in the manuscript). This takes 1 min to run
* ML_pipeline_binary.py and ML_pipeline_multi.py: produces multiple csv files containing the value for each run and the mean and standard deviation over 30 runs of the following performance metrics: AUC, accuracy, sensitivity, specificity, Cohen's Kappa score and precision along. It also saves the pre-processed data in a pickle format and the selected features in a csv format. This takes 20-30 min to run.
* GSManalysis_strainspecific.py: produces a list of the unique locus tags for the genes of interest in the selcted model (GSMGenesClinicalSampleID.csv and GSMGenesLineageSampleID.csv'; a list of genes essential in rich media (clinicalessentialinrich_SampleID.csv and lineageessentialinrich_SampleID.csv); a list of genes essential only in minimal media (clinicalessentialinminimal_SampleID.csv and lineageessentialinminimal_SampleID.csv), a list of the auxotrophic behaviour predicted in the model (SampleID_auxotrophy_clinical.csv and SampleID_auxotrophy_lineage.csv); a list of the growth rates on alternative carbon sources (SampleID_alternativecarbon_clinical.csv and SampleID_alternativecarbon_lineage.csv); a list of significantly changed reaction fluxes predicted as the result of gene knockouts (SignificantFVAreactions_SampleID.csv); a list of genes with significantly changed reaction fluxes predicted as the result of gene knockouts (SignificantFVAgenes_SampleID.csv); a list of significantly changed metabolite yields as the result of gene knockouts (SignificantFBAmetabolites_SampleID.csv); a list of genes with significantly changed metabolite yields as the result of gene knockouts (SignificantFBAgenes_SampleID.csv).This takes 20-30 min to run.
* GSM_VC960_clinical.ipynb: produces: a list of significant genes from FVA analysis (FVA.csv) All other outputs are displayed in console. This takes approximately 12-24 hours to run.
* GSM_VC960_lineage.ipynb: produces: a list of significant genes from FVA analysis (FVA.csv) All other outputs are displayed in console. This takes approximately 12-24 hours to run.

# Algorithm's Flow (ML_pipeline_binary.py and ML_pipeline_multi.py)

# License

This project is covered under the **AGPL-3.0 license**.

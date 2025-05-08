# PlantDRP
# Overview
This repository contains R code to run plantDRP tool from command line or R interface. A test file is provided for users to run and validate this script.

# Requirements
R 4.0.2 version (The program runs only in the R version 4.0.2, no older or latest version will work)
## R packages
### From CRAN
    xgboost
    magrittr
    Matrix
    PSSMCOOL
    R2HTML
    
### From Bioconductor
    Biostrings

# Download files and folder
- plantdrp.R
- example.fasta
- train.phr
- train.pin
- train.psq
- xgb.model
- drg_xgboost_feature.txt
- bin
- dataset

# File description
### plantdrp.R
It is an R file which contains the code to run the PlantDRP model from command line or R interface
### example.fasta
It is an example FASTA file consists of a header line starting with a ">" followed by the amino acid sequence
### train.phr
Contains the actual sequence data (nucleotide or protein) in a compressed, indexed format
### train.pin
Stores the sequence IDs from the original FASTA file, allowing BLAST to retrieve sequence names and descriptions
### train.psq
Stores the sequence quality information, such as the length and coverage of the sequences
### xgb.model
Our trained model with Extreme Boosting (XGB) algorithm
### drg_xgboost_feature.txt
contains the index of features which will be used for the prediction
### bin folder
contains psiblast executable
### dataset folder
contains the training dataset, which was used to train the model, and the testing dataset, which was used to evaluate the model's performance.

# Usage 
- Create a working directory.
- Place all the downloaded files and bin folder in the working directory.
- Before running the script the user needs to make few changes in the _**plantdrp.R**_ file.
  
  - User can use its own FASTA file by replacing the _**gene.fasta**_ file inside the R code with the name of its file in the following line
  
        [21] x <- readAAStringSet("example.fasta")
        [21] x <- readAAStringSet("replace with your FASTA file name")

  -  User have to set path according to there working directory in the following line

         [51] shell("path/to/working/directory/bin/psiblast -query path/to/working/directory/query.fasta -db path/to/working/directory/train -num_iterations 3 -evalue 0.001 -out_ascii_pssm path/to/working/directory/protein.pssm")

      Example:
  
         [51] shell("C:/xampp/htdocs/plantdrp/server/bin/psiblast -query C:/xampp/htdocs/plantdrp/server/query.fasta -db C:/xampp/htdocs/plantdrp/server/train -num_iterations 3 -evalue 0.001 -out_ascii_pssm C:/xampp/htdocs/plantdrp/server/protein.pssm")
  
- Running script
  
      path/to/R-4.0.2/Rscript <script.R>

  Example:
  
      C:/PROGRA~1/R/R-4.0.2/bin/Rscript plantdrp.R
   
   NOTE: User needs to give absolute path for R version 4.0.2 only

# Output description
After the program is completed, an HTML file will be generated that includes a table with the serial number in the first column, the amino acid sequence ID in the second, the type of protein (whether disease resistant or not) in the third, and the last column contains the probability by which the prediction has been made.

For example:

| |Sequence ID |Predicted as|With Probability|
|---|---|---|---|
|1|LOC_Os06g41640.1|DRP|1.00|
|2|Pl17|DRP|1.00|
|3|sp\|Q2TSC7\|PCS1_LOTJA|non-DRP|1.00|
|4|sp\|Q5SCW3\|YCF2_HUPLU|non-DRP|0.99|

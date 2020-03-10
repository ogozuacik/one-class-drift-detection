# Concept Learning using One-Class Classifiers for Implicit Drift Detection in Evolving Data Streams

**Parameters:**
* nu: parameter for SVM (set it to 0.5)
* size: window size
* percent: threshold for outlier percentage

**Command line instructions:**

* python OCDD.py dataset_name nu size percent (sample: python OCDD.py elec.csv 0.5 100 0.3)

* You can either put the datasets into the same directory or write dataset directory in place of dataset_name.
Datasets should be in **csv** format. You can access the datasets used in the paper and more from:
  * https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow

* You have to install scikit-multiflow in addition to commonly used python libraries. (sklearn, pandas, numpy, matplotlib)
  * https://scikit-multiflow.github.io/

**The code will output:** 
* Final accuracy
* Total elapsed time (from beginning of the stream to the end)
* Prequential accuracy plot (dividing data stream into 30 chunks)

# MSGGSA
There are some caveats to running this code:
1、The algorithm employed in this study is a wrapper feature selection algorithm specifically designed for cancer classification. Given the high dimensionality of the cancer gene dataset, it is recommended to apply a filtering method prior to executing this code in order to reduce computational costs. 
2、To illustrate, only a subset of the complete dataset is provided here as an example. The entire dataset can be obtained from https://portal.gdc.cancer.gov/ and https://xenabrowser.net/datapages/. Once the raw dataset has been downloaded from a public database, preprocessing steps such as removing rows and columns with more than 50 percent missing values should be performed. 
4、Finally, the filtered data can be input into MSGGSA to obtain accurate results.
# ROBL
1、This framework belongs to the two-population recursive framework, in which the idea of opposition-based learning is introduced to generate opposing populations. MSGGSA needs to be applied separately in the two populations, and the obtained feature Spaces of elite individuals and parate individuals are combined as the feature space of the next round of recursion. 
2、 This framework needs to be used in conjunction with the MSGGSA algorithm, which involves the code and data sets can be found in the MSGGSA file.

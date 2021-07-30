## Wikipedia preprocessing code
This directory contains the code to preprocess Wikipedias.    
First you need to download the Wikipedia dumps following [1. Download Wikipedia dumps](#1-download-wikipedia-dumps), preprocess and store the data into a sqlite DB file ([2. Store data into database](#2-store-data-into-database)), and then create a context file by splitting each article into 100 token long and write to a tsv file ([3. Create a DPR context file](#3-create-a-dpr-context-file)).

### 1. Download Wikipedia dumps
First, you need to download Wikipedia dump from [the Wikimedia website](https://dumps.wikimedia.org/). They only keep the most recent dumps, so if you are looking for dumps from certain timestamps, you have to check [the archive](https://archive.org/details/wikimediadownloads).

e.g., all of the related dump for Japanese Wikipedia 20190201 can be seen and downloaded [here](https://archive.org/download/jawiki-20190201). `jawiki-20190201-pages-articles-multistream.xml.bz2` includes the article text. 

### Run Wikiextractor to extract plain text 
We usually run [Wikiextractor](https://github.com/attardi/wikiextractor) to preprocess and extract plain text data from the Wikipedia dump. 

```
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
python WikiExtractor.py /path/to/your/xxwiki-20190201-pages-articles-multistream.xml.bz2 --filter_disambig_pages --json -o /path/to/output/directory -s
```

you can add `-c` (`--compress`) option to compress the output files using bzip. 

### 2. Store data into database 
You can store the processed text data into sqlite database. 

```
python build_db.py /path/to/preprocessed/data/dir /path/to/db/file.db
```

### 3. Create a DPR context file
DPR first splits each article into 100-token length instead of using the original paragraphs or articles as is. Run the command below to generate a tsv file where each line contains 100-token length Wikipedia paragraphs. 

```
python build_dpr_w100_data.py --db_path /path/to/db/file.db --tsv_path /path/to/output/file.tsv
```

Japanese and Thai does not use white spaces for segmentation. For those language, you need to run the special scripts below, which tokenize the input sequences and generate 100-token document chunks as in other languages.

- For Japanese: [create_w100_data_japanese.py](create_w100_data_japanese.py)
- For Thai: [create_w100_data_thai.py](create_w100_data_thai.py])

## References
- [List of Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias): you can check the statistics of each Wikipedia from the **Details table** section.
- [Wikimedia Archive](https://archive.org/details/wikimediadownloads?and[]=year%3A%222019%22)

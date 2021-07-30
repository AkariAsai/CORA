## BM25 baseline - index construction

### Installation

- Please follow the [guide](https://github.com/castorini/pyserini) to install Pyserini library in development 
installation section.
- Please follow the [guide](https://github.com/attardi/wikiextractor) to install WikiExtractor.

### Download the Wikipedia dump and cut the wikidump into paragraphs

- Download the Wikipedia database dump for a certain language `<lang>` at 
  `https://archive.org/details/<lang>wiki-20190201`
- run 
  ```shell
  python -m wikiextractor.WikiExtractor <INPUT PATH TO THE WIKI DUMP> --json -o <OUTPUT PATH>
  ```
  to extract the wikipedia dump data
- run 
```shell
python edit_json_file.py --input-dir <OUTPUT PATH OF THE LAST COMMAND> --output-dir <NEW OUTPUT PATH> --step <STEP SIZE>
```
to cut the wikipedia into chunks of `<STEP SIZE> ` paragraph, and the document id will be renamed as `<ORIG ID>-0`, 
`<ORIG ID>-<STEP SIZE>`, `<ORIG ID>-<STEP SIZE*2>`...

- to index the documents, do
```shell
python -m pyserini.index -collection JsonCollection -language ${LANGUAGE} -generator DefaultLuceneDocumentGenerator 
-threads <NUM-THREAD>  -input <OUTPUT PATH OF THE PREV STEP> -index <PLACE TO STORE THE INDEX> -storePositions 
 -storeDocvectors -storeRaw
```
- to search the documents and output the answer in SQUAD format, do
```shell
python search_index.py --index-path <PLACE TO STORE THE INDEX> --query-path <QUERY PATH> --output <OUTPUT PATH> 
--lang ${LANGUAGE}
```

For more details, please refer to each file.







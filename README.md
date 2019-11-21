# POS Tagger
This POS tagger is implemented in HMM with Viterbi decoding algorithm.

## To run
```sh
$ python POS_Tagger.py <training file> <file to tag> <output file> <ngram> <method to handle unknown words>
```

### Use included scorer to evaluate performance
```sh
$ python scorer.py POS_dev.pos <your result>
```

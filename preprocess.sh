BART_PATH="$1"
TASK="$2"

rm -rf "${TASK}-bin/"
fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "${TASK}/train.bpe" \
    --validpref "${TASK}/val.bpe" \
    --testpref "${TASK}/test.bpe" \
    --destdir "${TASK}-bin/" \
    --workers 60 \
    --srcdict "${BART_PATH}/dict.txt" \
    --tgtdict "${BART_PATH}/dict.txt";
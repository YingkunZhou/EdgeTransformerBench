FMT=$1
DEV=$2
NOSLEEP=0
SLEEP="${3:-$NOSLEEP}"
EVAL=$4
DATA=$5

if [ ! -d ".coreml" ]
then
    [ ! -f "coreml-models.tar.gz" ] && gdown 18oaKtzRDyaHnmXe2ugKCCZ8dWAGsD8OC
    tar xf coreml-models.tar.gz;
fi

python python/coreml-perf.py --format $FMT --compute $DEV --sleep $SLEEP  $EVAL $DATA --only-test ALL 2>/dev/null
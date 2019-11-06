#!/usr/bin/env bash
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0

python baseline/training.py \
    --cuda \
    --src en \
    --tgt nso \
    --model-file en-nso-baseline.pt \
    --n-layers 3 \
    --n-heads 4 \
    --embed-dim 512 \
    --hidden-dim 512 \
    --dropout 0.2 \
    --word-dropout 0.1 \
    --lr 5e-4 \
    --n-epochs 30 \
    --tokens-per-batch 4000 \
    --clip-grad 1.0

python baseline/translate.py \
    --cuda \
    --src en \
    --tgt nso \
    --model-file en-nso-baseline.pt \
    --search "beam_search" \
    --beam-size 5 \
    --input-file ../data/en_nso/ennso_parallel.bpe.dev.en \
    --output-file ennso_parallel.dev.out.nso

python baseline/translate.py \
    --cuda \
    --src en \
    --tgt nso \
    --model-file en-nso-baseline.pt \
    --search "beam_search" \
    --beam-size 5 \
    --input-file ../data/en_nso/ennso_parallel.bpe.test.en \
    --output-file ennso_parallel.test.out.nso

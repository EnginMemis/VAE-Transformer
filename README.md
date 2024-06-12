# VAE-Transformer
GENERATING PARAPHRASED SENTENCES USING UNSUPERVISED LEARNING WITH VARIATIONAL AUTOENCODER MODELS

## FOLDER ORGANIZATION

      data---
            |
            -- corpus.train.txt
            |
            -- corpus.valid.txt

## MODEL TRAINING

```bash
python trainer.py \
       --data_name "corpus" \
       --data_dir "data" \ 
       --epochs 3 \
       --learning_rate 0.001 \
       --batch_size 64
```

## AUGMENTATION

```bash
python augment.py \
       --data_name "{data_name}" \
       --save_model_path "models" \ 
       --checkpoint "{model_path}" \
       ---max_sentence_length 60 \
       --rate 0.5
```

## DEMO

```bash
python demo.py \
       --checkpoint "{model_path}" \
       --data_name "{data_name}" \ 
       --save_model_path "models" \
       --noise "False" \
```


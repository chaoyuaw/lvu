
export TRAIN_FILE=data/ava/train.csv
export TEST_FILE=data/ava/val.csv

export TRAIN_FEAT_FILE=data/ava/train_features.pkl
export TEST_FEAT_FILE=data/ava/val_features.pkl


exp=`date +"%Y%m%d_%H%M%S"`_ava


short_term_model_weights=data/ava/SLOWFAST_32x2_R101_50_50.pkl
pretrained_weights=/u/cywu/cywu/lvu_release_dev/pretrained_models/pretrained_for_ava.bin


python -u src/run.py \
    --output_dir=outputs/${exp} \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --do_eval \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --train_feature_file=$TRAIN_FEAT_FILE \
    --eval_feature_file=$TEST_FEAT_FILE \
    --mlm \
    --evaluate_during_training \
    --exp ${exp} \
    --num_train_epochs 300 \
    --learning_rate 1e-4 \
    --warmup_steps 0 \
    --per_gpu_train_batch_size 32 \
    --num_workers 8 \
    --num_workers_eval 8 \
    ${in_args} \
    --weight_decay 0.01 \
    --save_total_limit 0 \
    --eval_epochs 300 \
    --save_steps 0 \
    --action_recognition \
    --mask_sep_no_mask \
    --train_long_term_linear \
    --train_long_term_dropout \
    --per_gpu_eval_batch_size 32 \
    --short_term_model_weights ${short_term_model_weights} \
    --force_load_checkpoint ${pretrained_weights} \
    >> logs/${exp}.log 2>&1




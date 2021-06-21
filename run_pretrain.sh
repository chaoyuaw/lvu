

export TEST_FILE=data/ava/val.csv
export TEST_FEAT_FILE=data/ava/val_features.pkl


exp=`date +"%Y%m%d_%H%M%S"`_pretrain


python -u src/run.py \
    --output_dir=outputs/${exp} \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --do_eval \
    --mc_train_feature_file="data/features/" \
    --train_data_file=@@@ \
    --eval_data_file=$TEST_FILE \
    --train_feature_file=@@@ \
    --eval_feature_file=$TEST_FEAT_FILE \
    --mlm \
    --evaluate_during_training \
    --exp ${exp} \
    --num_train_epochs 2.0 \
    --learning_rate 1e-4 \
    --warmup_steps 3000 \
    --per_gpu_train_batch_size 16 \
    --num_workers 8 \
    --num_workers_eval 2 \
    --weight_decay 0.01 \
    --save_total_limit 1 \
    --seed 42 \
    --action_loss_weight 80.0 \
    --use_soft_labels \
    --temperature 1.0 \
    --mask_sep \
    --same_movie \
    --same_movie_temperature 0.2 \
    --same_movie_weight 1.0 \
    --logging_steps 50 \
    --long_term_task_name pretrain_demo \

   # >> logs/${exp}.log 2>&1



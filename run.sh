
if (( $1 == 1 ))
then
    long_term_task_name=relationship
    num_long_term_classes=4
fi
if (( $1 == 2 ))
then
    long_term_task_name=way_speaking
    num_long_term_classes=5
fi

if (( $1 == 3 ))
then
    long_term_task_name=scene
    num_long_term_classes=6
fi
if (( $1 == 4 ))
then
    long_term_task_name=director
    num_long_term_classes=10
fi
if (( $1 == 5 ))
then
    long_term_task_name=writer
    num_long_term_classes=10
fi
if (( $1 == 6 ))
then
    long_term_task_name=year
    num_long_term_classes=9
fi
if (( $1 == 7 ))
then
    long_term_task_name=genre
    num_long_term_classes=4
fi
if (( $1 == 8 ))
then
    long_term_task_name=like_ratio
    num_long_term_classes=-1
fi
if (( $1 == 9 ))
then
    long_term_task_name=view_count
    num_long_term_classes=-1
fi

exp=`date +"%Y%m%d_%H%M%S"`_${long_term_task_name}


####################

in_args="--force_load_checkpoint pretrained_models/mask_compact.bin"


python -u src/run.py \
    --output_dir=outputs/${exp} \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --do_eval \
    --mc_train_feature_file="data/features/" \
    --train_data_file=@@@ \
    --eval_data_file=@@@ \
    --train_feature_file=@@@ \
    --eval_feature_file=@@@ \
    --mlm \
    --evaluate_during_training \
    --exp ${exp} \
    --num_train_epochs 10 \
    --eval_epochs 10 \
    --learning_rate 2e-5 \
    --warmup_steps 0 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --num_workers 8 \
    --num_workers_eval 8 \
    ${in_args} \
    --weight_decay 0.01 \
    --save_total_limit 0 \
    --save_steps 0 \
    --use_good_quality \
    --mask_sep_no_mask \
    --train_long_term_linear \
    --train_long_term_dropout \
    --three_split \
    --use_soft_labels \
    --temperature 1.0 \
    --train_long_term \
    --long_term_task_name ${long_term_task_name} \
    --num_long_term_classes ${num_long_term_classes} \
    --mask_sep \

   # >> logs/${exp}.log 2>&1

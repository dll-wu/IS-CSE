export CUDA_VISIBLE_DEVICES=0
python -u train.py \
    --model_name_or_path bert-large-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/iscse-bert-large-cos-alpha-0.005-0.05 \
    --alpha_mode_dynamic \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --overwrite_cache \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

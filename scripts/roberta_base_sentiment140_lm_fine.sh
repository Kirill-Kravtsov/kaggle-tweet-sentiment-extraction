python ../../transformers/examples/language-modeling/run_language_modeling.py \
	--output_dir=../pretrained/roberta-base-sentiment140-lm-fine \
	--model_type=roberta \
	--model_name_or_path=../pretrained/roberta-base/ \
	--do_train \
	--train_data_file=../data/external/sentiment140_lm.txt \
	--mlm \
	--overwrite_cache \
	--line_by_line \
	--overwrite_output_dir

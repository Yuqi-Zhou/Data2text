
# 1.preprocess

BASE=rotowire
PART=dataset
KEY=roto

python preprocess.py \
-train_src1 $BASE/$PART/src_train.txt \
-train_tgt1 $BASE/$PART/train_content_plan.txt \
-train_src2 $BASE/$PART/inter/train_content_plan.txt \
-train_tgt2 $BASE/$PART/tgt_train.txt \
-train_ptr_rsrc $BASE/$PART/train-${KEY}-ptrs-input.txt \
-train_tgt2_ptrs $BASE/$PART/tgt_ptrs_idx_train.txt \
-train_ptr $BASE/$PART/train-${KEY}-ptrs.txt \
-valid_src1 $BASE/$PART/src_valid.txt \
-valid_tgt2_ptrs $BASE/$PART/tgt_ptrs_idx_valid.txt \
-valid_tgt1 $BASE/$PART/valid_content_plan.txt \
-valid_src2 $BASE/$PART/inter/valid_content_plan.txt \
-valid_tgt2 $BASE/$PART/tgt_valid.txt \
-save_data $BASE/preprocess/${KEY} \
-src_seq_length 1000 -dynamic_dict -tgt_seq_length 1000

# 2.train

BASE=rotowire
IDENTIFIER=cc
GPUID=0

python3 train.py -data $BASE/preprocess/roto \
-save_model $BASE/gen_model/ncpcc/roto \
-encoder_type1 mean -decoder_type1 pointer \
-enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn \
-decoder_type2 rnn -enc_layers2 2 -dec_layers2 2 \
-batch_size 5 -feat_merge mlp -feat_vec_size 600 \
-word_vec_size 600 -rnn_size 600 -seed 1234 \
-start_checkpoint_at 2 -epochs 100 -optim adagrad \
-learning_rate 0.15 -adagrad_accumulator_init 0.1 \
-report_every 100 -copy_attn -truncated_decoder 100 \
-gpuid $GPUID -attn_hidden 64 -reuse_copy_attn \
-start_decay_at 4 -learning_rate_decay 0.97 \
-valid_batch_size 5 >logs/log.train.ncpcc3

# 3.generate content plan

BASE=rotowire
PART=dataset
IDENTIFIER=cc
GPUID=1

MODEL_PATH=$BASE/gen_model/ncpcc/roto_stage1_acc_66.8511_ppl_1.0074_e10.pt
MODEL_PATH2=$BASE/gen_model/ncpcc/roto_stage2_acc_47.3817_ppl_8.0361_e16.pt

python3 translate.py \
-model $MODEL_PATH \
-src1 $BASE/dataset/inf_src_valid.txt \
-output $BASE/$PART/valid/gen/roto_stage1_$IDENTIFIER-beam5_gens.txt \
-batch_size 10 -max_length 80 -gpu $GPUID -min_length 35 -stage1


# 4. generate content plan with records
python3 create_content_plan_from_index.py \
$BASE/dataset/inf_src_valid.txt \
$BASE/$PART/valid/gen/roto_stage1_$IDENTIFIER-beam5_gens.txt \
$BASE/$PART/valid/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.h5-tuples.txt  \
$BASE/$PART/valid/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.txt

# 5. calculate accuracy of content plan in first stage
python3 non_rg_metrics.py ${BASE}_h5/roto-gold-val.h5-tuples.txt \
$BASE/$PART/valid/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.h5-tuples.txt

# 6. output summary
python3 translate.py \
-model $MODEL_PATH -model2 $MODEL_PATH2 \
-src1 $BASE/dataset/inf_src_valid.txt \
-tgt1 $BASE/$PART/valid/gen/roto_stage1_$IDENTIFIER-beam5_gens.txt \
-src2 $BASE/$PART/valid/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.txt \
-output $BASE/$PART/valid/gen/roto_stage2_$IDENTIFIER-beam5_gens.txt \
-batch_size 1 -max_length 850 -min_length 150 -gpu $GPUID

# 7. Metrics of RG, CS, CO

# # 7.1 把train.json/valid.json等转成h5格式
# 7.1 把生成的valid集的summary转换成IE model消费的数据结构
python3 data_utils.py -mode prep_gen_data \
-gen_fi $BASE/$PART/valid/gen/roto_stage2_$IDENTIFIER-beam5_gens.txt \
-dict_pfx "${BASE}_h5/roto-ie" \
-output_fi $BASE/$PART/valid/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.h5 \
-input_path $BASE/dataset

# 7.2 采用IE model在valid集的summary h5中进行关系抽取，并计算RG
BASE=rotowire
PART=dataset
IDENTIFIER=cc
KEY=roto
GPUID=0

python3 extractor.py -datafile ${BASE}_h5/roto-ie.h5 \
-preddata $BASE/$PART/valid/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.h5 \
-dict_pfx "${BASE}_h5/roto-ie" -gpuid $GPUID \
-savefile ie_model/lstm/lstm -epochs 10 -lstm

python3 extractor.py -datafile ${BASE}_h5/roto-ie.h5 \
-preddata $BASE/$PART/valid/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.h5 \
-dict_pfx "${BASE}_h5/roto-ie" -gpuid $GPUID \
-savefile ie_model/conv/conv -epochs 10 -lr 0.5

python3 extractor.py -datafile ${BASE}_h5/roto-ie.h5 \
-preddata $BASE/$PART/valid/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.h5 \
-dict_pfx "${BASE}_h5/roto-ie" -gpuid $GPUID \
-just_eval

# 7.3 CS, CO(DLD)

python3 non_rg_metrics.py ${BASE}_h5/roto-gold-val.h5-tuples.txt \
$BASE/$PART/valid/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.h5-tuples.txt

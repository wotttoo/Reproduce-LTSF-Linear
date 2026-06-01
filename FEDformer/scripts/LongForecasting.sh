# cd FEDformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

for preLen in 96 192 336 720
do

# exchange
python -u run.py \
 --is_training 1 \
 --root_path ../dataset/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 1 >../logs/LongForecasting/FEDformer_exchange_rate_${preLen}.log
done

# cd ..

# cd Pyraformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

# Exchange
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256  >../logs/LongForecasting/Pyraformer_exchange_rate_96.log
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256  >../logs/LongForecasting/Pyraformer_exchange_rate_192.log
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 336 -n_head 6 -lr 0.00001 -d_model 256  >../logs/LongForecasting/Pyraformer_exchange_rate_336.log
python long_range_main.py  -data_path exchange_rate.csv -data exchange \
-input_size 96 -predict_step 720 -n_head 6 -lr 0.00001 -d_model 256  >../logs/LongForecasting/Pyraformer_exchange_rate_720.log

# cd ..

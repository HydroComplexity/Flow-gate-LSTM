# Flow_gate-LSTM
Files overview: 

1. interdep_7hour_FGLSTM.py: Run to find the interdependency network among the solutes. Predictions at a one-time step ahead after initial training. It uses the observed historical data to predict.
2. FGLSTM_kweek.py: Run for the prediction for several weeks step ahead after initial training. It uses the predicted historical data to predict for the k-week.
3. lstm_classes.py: Contains the modified LSTM architecture
4. utils_fglstm.py: Contain functions used in the 1. and 2.  
5. plot_jk_incre.py: Plot figures for incremental model
6. plot_jk_noincre.py: Plot figures for prediction at 1 time step ahead
7. sd01.xlsx: Upper hafren, Plynimon, UK solute chemistry and river flow rate data, 7 hr frequency


python tool versions:
1. python: 3.10.13
2. numpy: 1.23.4
3. scipy: 1.10.1
4. pytorch: 1.13.0
5. pandas: 2.1.4

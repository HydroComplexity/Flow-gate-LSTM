# Flow-gate-LSTM
Summary: 
Solute concentrations and mass exports from rivers are important to understand and predict, as they impact water quality downstream. Solutes in streamflow include nitrates, calcium, sodium, magnesium, and other cations and anions, and concentrations depend on watershed characteristics such as vegetation, soil, and bedrock composition, flow paths and residence times, microbial activity, water quantity, and human applications of road salts or fertilizers. In other words, solute inputs to rivers vary with time, discharge, and source of water, and solute dynamics are often interrelated due to common sources. This, combined with a few high-resolution observations of stream solutes makes predictions of solute concentrations challenging. We build on a machine learning framework, LSTM, to improve solute predictions in two ways. First, we add to the traditional LSTM model architecture to specifically incorporate flow gradient into the model, to better capture hysteresis, where concentrations vary depending on whether flow is increasing or decreasing. Second, to predict a given solute, we detect a set of highly informative solutes, that best predict the target solute without adding redundancies in model inputs. These changes in model input selection and architecture lead to large model improvements, for both long-term (several weeks) and short-term (7 hours) predictions. 

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

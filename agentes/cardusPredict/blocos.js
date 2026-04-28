const blockDB = {
  'CSV Input': { type:'data', inputs:[], outputs:['tabular_data'], logic:'CSV', function:'Leitura', synergies:['Normalization','Data Cleaning'], incompatibilities:[], desc:'Importa CSV' },'TXT Input': { 
    type: 'data', 
    inputs: [], 
    outputs: ['raw_text', 'tabular_data'], 
    logic: 'TXT', 
    function: 'Leitura', 
    synergies: ['Regex Extraction', 'Data Cleaning', 'Tokenization'], 
    incompatibilities: [], 
    desc: 'Importa arquivos de texto plano (txt)' 
},
  'JSON Input': { type:'data', inputs:[], outputs:['json_data'], logic:'JSON', function:'Leitura', synergies:['Data Parsing'], incompatibilities:[], desc:'Importa JSON' },
  'XML Input': { type:'data', inputs:[], outputs:['xml_data'], logic:'XML', function:'Leitura', synergies:['Data Parsing'], incompatibilities:[], desc:'Importa XML' },
  'Parquet Input': { type:'data', inputs:[], outputs:['columnar_data'], logic:'Parquet', function:'Leitura', synergies:['Feature Selection'], incompatibilities:[], desc:'Arquivo Parquet' },
  'Excel Input': { type:'data', inputs:[], outputs:['spreadsheet_data'], logic:'Excel', function:'Leitura', synergies:['Data Cleaning'], incompatibilities:[], desc:'Planilha Excel' },
  'API REST': { type:'data', inputs:[], outputs:['api_data'], logic:'REST', function:'Coleta', synergies:['Data Parsing'], incompatibilities:[], desc:'API REST' },
  'GraphQL API': { type:'data', inputs:[], outputs:['graphql_data'], logic:'GraphQL', function:'Coleta', synergies:['Data Parsing'], incompatibilities:[], desc:'GraphQL' },
  'Database SQL': { type:'data', inputs:[], outputs:['sql_data'], logic:'SQL', function:'Leitura', synergies:['Data Cleaning'], incompatibilities:[], desc:'Banco SQL' },
  'MongoDB': { type:'data', inputs:[], outputs:['nosql_data'], logic:'NoSQL', function:'Leitura', synergies:['Data Parsing'], incompatibilities:[], desc:'MongoDB' },
  'BigQuery': { type:'data', inputs:[], outputs:['bigdata'], logic:'BigQuery', function:'Leitura', synergies:['Aggregation'], incompatibilities:[], desc:'Google BigQuery' },
  'AWS S3': { type:'data', inputs:[], outputs:['cloud_data'], logic:'S3', function:'Leitura', synergies:['Data Cleaning'], incompatibilities:[], desc:'AWS S3' },
  'Google Cloud Storage': { type:'data', inputs:[], outputs:['cloud_data'], logic:'GCS', function:'Leitura', synergies:['Data Cleaning'], incompatibilities:[], desc:'GCS' },
  'Azure Blob': { type:'data', inputs:[], outputs:['cloud_data'], logic:'Azure', function:'Leitura', synergies:['Data Cleaning'], incompatibilities:[], desc:'Azure Blob' },
  'Kafka Stream': { type:'data', inputs:[], outputs:['stream_data'], logic:'Kafka', function:'Streaming', synergies:['Real-time Processing'], incompatibilities:[], desc:'Apache Kafka' },
  'Web Scraping': { type:'data', inputs:[], outputs:['scraped_data'], logic:'Scrape', function:'Coleta', synergies:['Data Cleaning'], incompatibilities:[], desc:'Web Scraping' },
  'PDF Extract': { type:'data', inputs:[], outputs:['pdf_text'], logic:'PDF', function:'Extração', synergies:['Text Processing'], incompatibilities:[], desc:'Extrai texto de PDF' },
  'Image Dataset': { type:'data', inputs:[], outputs:['image_data'], logic:'Image', function:'Leitura', synergies:['CNN','Image Preprocessing'], incompatibilities:[], desc:'Conjunto de imagens' },
  'Video Dataset': { type:'data', inputs:[], outputs:['video_data'], logic:'Video', function:'Leitura', synergies:['Video Preprocessing'], incompatibilities:[], desc:'Conjunto de vídeos' },
  'Audio Dataset': { type:'data', inputs:[], outputs:['audio_data'], logic:'Audio', function:'Leitura', synergies:['Audio Preprocessing'], incompatibilities:[], desc:'Conjunto de áudio' },
  'Text Dataset': { type:'data', inputs:[], outputs:['text_data'], logic:'Text', function:'Leitura', synergies:['NLP','Tokenization'], incompatibilities:[], desc:'Conjunto de textos' },
  'Time Series Data': { type:'data', inputs:[], outputs:['timeseries_data'], logic:'TimeSeries', function:'Leitura', synergies:['Detrending','RNN','LSTM'], incompatibilities:[], desc:'Séries temporais' },
  'Graph Data': { type:'data', inputs:[], outputs:['graph_data'], logic:'Graph', function:'Leitura', synergies:['GNN'], incompatibilities:[], desc:'Dados em grafo' },
  'Synthetic Data Generator': { type:'data', inputs:[], outputs:['synthetic_data'], logic:'Synthetic', function:'Geração', synergies:['Testing'], incompatibilities:[], desc:'Gera dados sintéticos' },
  'Normalization': { type:'preprocessing', inputs:['tabular_data'], outputs:['normalized_data'], logic:'MinMax', function:'Escala', synergies:['Linear Regression','Neural Network'], incompatibilities:[], desc:'Normalização 0-1' },
  'Standardization': { type:'preprocessing', inputs:['tabular_data'], outputs:['standardized_data'], logic:'ZScore', function:'Escala', synergies:['SVM','KNN'], incompatibilities:[], desc:'Padronização Z-score' },
  'Robust Scaling': { type:'preprocessing', inputs:['tabular_data'], outputs:['robust_scaled'], logic:'Robust', function:'Escala', synergies:['Outlier Detection'], incompatibilities:[], desc:'Escala robusta a outliers' },
  'Log Transformation': { type:'preprocessing', inputs:['tabular_data'], outputs:['log_data'], logic:'Log', function:'Transformação', synergies:['Linear Regression'], incompatibilities:[], desc:'Transformação logarítmica' },
  'Data Cleaning': { type:'preprocessing', inputs:['tabular_data'], outputs:['clean_data'], logic:'Clean', function:'Limpeza', synergies:['Feature Selection'], incompatibilities:[], desc:'Limpeza de dados' },
  'Missing Value Imputation': { type:'preprocessing', inputs:['tabular_data'], outputs:['imputed_data'], logic:'Impute', function:'Imputação', synergies:['Data Cleaning'], incompatibilities:[], desc:'Imputação de faltantes' },
  'Outlier Detection': { type:'preprocessing', inputs:['tabular_data'], outputs:['outlier_flagged'], logic:'Outlier', function:'Detecção', synergies:['Robust Scaling'], incompatibilities:[], desc:'Detecta outliers' },
  'Feature Selection': { type:'preprocessing', inputs:['tabular_data'], outputs:['selected_features'], logic:'Select', function:'Redução', synergies:['Random Forest','Linear Regression'], incompatibilities:[], desc:'Seleção de features' },
  'PCA': { type:'preprocessing', inputs:['tabular_data'], outputs:['pca_components'], logic:'PCA', function:'Redução', synergies:['Clustering'], incompatibilities:[], desc:'Análise de Componentes Principais' },
  't-SNE': { type:'preprocessing', inputs:['tabular_data'], outputs:['tsne_embedding'], logic:'tSNE', function:'Redução', synergies:['Visualization'], incompatibilities:[], desc:'t-SNE para visualização' },
  'UMAP': { type:'preprocessing', inputs:['tabular_data'], outputs:['umap_embedding'], logic:'UMAP', function:'Redução', synergies:['Visualization'], incompatibilities:[], desc:'UMAP' },
  'One-Hot Encoding': { type:'preprocessing', inputs:['tabular_data'], outputs:['encoded_data'], logic:'OneHot', function:'Codificação', synergies:['Neural Network'], incompatibilities:[], desc:'One-hot encoding' },
  'Label Encoding': { type:'preprocessing', inputs:['tabular_data'], outputs:['encoded_data'], logic:'Label', function:'Codificação', synergies:['Tree Models'], incompatibilities:[], desc:'Label encoding' },
  'Target Encoding': { type:'preprocessing', inputs:['tabular_data'], outputs:['encoded_data'], logic:'Target', function:'Codificação', synergies:['Gradient Boosting'], incompatibilities:[], desc:'Target encoding' },
  'Polynomial Features': { type:'preprocessing', inputs:['tabular_data'], outputs:['poly_features'], logic:'Poly', function:'Expansão', synergies:['Linear Regression'], incompatibilities:[], desc:'Features polinomiais' },
  'Binning': { type:'preprocessing', inputs:['tabular_data'], outputs:['binned_data'], logic:'Bin', function:'Discretização', synergies:['Tree Models'], incompatibilities:[], desc:'Discretização em bins' },
  'Tokenization': { type:'preprocessing', inputs:['text_data'], outputs:['tokens'], logic:'Token', function:'Texto', synergies:['NLP'], incompatibilities:[], desc:'Tokenização de texto' },
  'Stop Words Removal': { type:'preprocessing', inputs:['text_data'], outputs:['filtered_text'], logic:'StopWords', function:'Texto', synergies:['NLP'], incompatibilities:[], desc:'Remove stop words' },
  'Stemming': { type:'preprocessing', inputs:['text_data'], outputs:['stemmed_text'], logic:'Stem', function:'Texto', synergies:['NLP'], incompatibilities:[], desc:'Stemming' },
  'Lemmatization': { type:'preprocessing', inputs:['text_data'], outputs:['lemmatized_text'], logic:'Lemma', function:'Texto', synergies:['NLP'], incompatibilities:[], desc:'Lematização' },
  'TF-IDF Vectorization': { type:'preprocessing', inputs:['text_data'], outputs:['tfidf_vectors'], logic:'TFIDF', function:'Vetorização', synergies:['Text Classification'], incompatibilities:[], desc:'TF-IDF' },
  'Word2Vec': { type:'preprocessing', inputs:['text_data'], outputs:['word_embeddings'], logic:'W2V', function:'Embedding', synergies:['NLP'], incompatibilities:[], desc:'Word2Vec' },
  'BERT Embeddings1': { type:'preprocessing', inputs:['text_data'], outputs:['bert_embeddings'], logic:'BERT', function:'Embedding', synergies:['Transformer'], incompatibilities:[], desc:'Embeddings BERT' },
  'Image Resizing': { type:'preprocessing', inputs:['image_data'], outputs:['resized_images'], logic:'Resize', function:'Imagem', synergies:['CNN'], incompatibilities:[], desc:'Redimensiona imagens' },
  'Image Normalization': { type:'preprocessing', inputs:['image_data'], outputs:['norm_images'], logic:'NormImg', function:'Imagem', synergies:['CNN'], incompatibilities:[], desc:'Normalização de imagens' },
  'Image Augmentation': { type:'preprocessing', inputs:['image_data'], outputs:['augmented_images'], logic:'Augment', function:'Imagem', synergies:['CNN'], incompatibilities:[], desc:'Data augmentation' },
  'Grayscale Conversion': { type:'preprocessing', inputs:['image_data'], outputs:['grayscale'], logic:'Gray', function:'Imagem', synergies:['CNN'], incompatibilities:[], desc:'Conversão para escala de cinza' },
  'Detrending': { type:'preprocessing', inputs:['timeseries_data'], outputs:['detrended'], logic:'Detrend', function:'Série Temporal', synergies:['Time Series Analysis'], incompatibilities:[], desc:'Remove tendência' },
  'Differencing': { type:'preprocessing', inputs:['timeseries_data'], outputs:['stationary'], logic:'Diff', function:'Série Temporal', synergies:['ARIMA'], incompatibilities:[], desc:'Diferenciação' },
  'Lag Features': { type:'preprocessing', inputs:['timeseries_data'], outputs:['lag_features'], logic:'Lag', function:'Série Temporal', synergies:['LSTM'], incompatibilities:[], desc:'Cria features de lag' },
  'Rolling Window': { type:'preprocessing', inputs:['timeseries_data'], outputs:['rolling_stats'], logic:'Rolling', function:'Série Temporal', synergies:['Feature Engineering'], incompatibilities:[], desc:'Estatísticas em janela' },
  'Linear Regression': { type:'model', inputs:['normalized_data'], outputs:['regression_predictions'], logic:'Linear', function:'Regressão', synergies:['Normalization'], incompatibilities:['Random Forest'], desc:'Regressão Linear' },
  'Ridge Regression': { type:'model', inputs:['normalized_data'], outputs:['ridge_pred'], logic:'Ridge', function:'Regressão', synergies:['Normalization'], incompatibilities:[], desc:'Ridge Regression' },
  'Lasso Regression': { type:'model', inputs:['normalized_data'], outputs:['lasso_pred'], logic:'Lasso', function:'Regressão', synergies:['Normalization'], incompatibilities:[], desc:'Lasso Regression' },
  'Elastic Net': { type:'model', inputs:['normalized_data'], outputs:['elastic_pred'], logic:'ElasticNet', function:'Regressão', synergies:['Normalization'], incompatibilities:[], desc:'Elastic Net' },
  'Polynomial Regression': { type:'model', inputs:['poly_features'], outputs:['poly_pred'], logic:'PolyReg', function:'Regressão', synergies:['Polynomial Features'], incompatibilities:[], desc:'Regressão Polinomial' },
  'Support Vector Regression': { type:'model', inputs:['normalized_data'], outputs:['svr_pred'], logic:'SVR', function:'Regressão', synergies:['Normalization'], incompatibilities:[], desc:'SVR' },
  // Classificação
  'Logistic Regression': { type:'model', inputs:['normalized_data'], outputs:['classification'], logic:'Logistic', function:'Classificação', synergies:['Normalization'], incompatibilities:[], desc:'Regressão Logística' },
  'Softmax Regression': { type:'model', inputs:['normalized_data'], outputs:['multiclass'], logic:'Softmax', function:'Classificação', synergies:['Normalization'], incompatibilities:[], desc:'Softmax' },
  'Decision Tree': { type:'model', inputs:['clean_data'], outputs:['tree_pred'], logic:'Tree', function:'Classificação', synergies:['Feature Selection'], incompatibilities:[], desc:'Árvore de Decisão' },
  'Random Forest': { type:'model', inputs:['clean_data'], outputs:['ensemble_pred'], logic:'RF', function:'Classificação', synergies:['Feature Selection'], incompatibilities:['Linear Regression'], desc:'Random Forest' },
  'Gradient Boosting': { type:'model', inputs:['clean_data'], outputs:['gb_pred'], logic:'GB', function:'Classificação', synergies:['Feature Selection'], incompatibilities:[], desc:'Gradient Boosting' },
  'XGBoost': { type:'model', inputs:['clean_data'], outputs:['xgb_pred'], logic:'XGB', function:'Classificação', synergies:['Feature Selection'], incompatibilities:[], desc:'XGBoost' },
  'LightGBM': { type:'model', inputs:['clean_data'], outputs:['lgb_pred'], logic:'LGB', function:'Classificação', synergies:['Feature Selection'], incompatibilities:[], desc:'LightGBM' },
  'CatBoost': { type:'model', inputs:['clean_data'], outputs:['cat_pred'], logic:'CatBoost', function:'Classificação', synergies:['Feature Selection'], incompatibilities:[], desc:'CatBoost' },
  'K-Nearest Neighbors': { type:'model', inputs:['normalized_data'], outputs:['knn_pred'], logic:'KNN', function:'Classificação', synergies:['Normalization'], incompatibilities:[], desc:'KNN' },
  'Support Vector Machine': { type:'model', inputs:['normalized_data'], outputs:['svm_pred'], logic:'SVM', function:'Classificação', synergies:['Normalization'], incompatibilities:[], desc:'SVM' },
  'Naive Bayes': { type:'model', inputs:['clean_data'], outputs:['nb_pred'], logic:'NB', function:'Classificação', synergies:['Feature Selection'], incompatibilities:[], desc:'Naive Bayes' },
  // Redes Neurais
  'Neural Network': { type:'model', inputs:['normalized_data'], outputs:['nn_pred'], logic:'NN', function:'Modelagem', synergies:['Normalization'], incompatibilities:[], desc:'Rede Neural MLP' },
  'CNN': { type:'model', inputs:['image_data'], outputs:['cnn_pred'], logic:'CNN', function:'Visão Computacional', synergies:['Image Preprocessing'], incompatibilities:[], desc:'Rede Neural Convolucional' },
  'RNN': { type:'model', inputs:['timeseries_data'], outputs:['rnn_pred'], logic:'RNN', function:'Sequencial', synergies:['Time Series Analysis'], incompatibilities:[], desc:'Rede Neural Recorrente' },
  'LSTM': { type:'model', inputs:['timeseries_data'], outputs:['lstm_pred'], logic:'LSTM', function:'Sequencial', synergies:['Time Series Analysis'], incompatibilities:[], desc:'Long Short-Term Memory' },
  'GRU': { type:'model', inputs:['timeseries_data'], outputs:['gru_pred'], logic:'GRU', function:'Sequencial', synergies:['Time Series Analysis'], incompatibilities:[], desc:'Gated Recurrent Unit' },
  'Transformer': { type:'model', inputs:['sequence_data'], outputs:['transformer_out'], logic:'Transformer', function:'Sequencial', synergies:['BERT Embeddings'], incompatibilities:[], desc:'Transformer' },
  'Autoencoder': { type:'model', inputs:['normalized_data'], outputs:['encoded_features'], logic:'AE', function:'Redução', synergies:['Anomaly Detection'], incompatibilities:[], desc:'Autoencoder' },
  'GAN': { type:'model', inputs:['noise_data'], outputs:['generated_data'], logic:'GAN', function:'Geração', synergies:['Image Data'], incompatibilities:[], desc:'Generative Adversarial Network' },
  // Clustering
  'K-Means': { type:'model', inputs:['normalized_data'], outputs:['clusters'], logic:'KMeans', function:'Clustering', synergies:['Normalization'], incompatibilities:[], desc:'K-Means' },
  'DBSCAN': { type:'model', inputs:['normalized_data'], outputs:['dbscan_clusters'], logic:'DBSCAN', function:'Clustering', synergies:['Normalization'], incompatibilities:[], desc:'DBSCAN' },
  'Hierarchical Clustering': { type:'model', inputs:['normalized_data'], outputs:['hierarchical'], logic:'Hierarchical', function:'Clustering', synergies:['Normalization'], incompatibilities:[], desc:'Clustering Hierárquico' },
  'Gaussian Mixture Model': { type:'model', inputs:['normalized_data'], outputs:['gmm_clusters'], logic:'GMM', function:'Clustering', synergies:['Normalization'], incompatibilities:[], desc:'GMM' },
  // Anomalia
  'Isolation Forest': { type:'model', inputs:['tabular_data'], outputs:['anomaly_scores'], logic:'IsoForest', function:'Anomalia', synergies:['Outlier Detection'], incompatibilities:[], desc:'Isolation Forest' },
  'One-Class SVM': { type:'model', inputs:['normalized_data'], outputs:['ocsvm_scores'], logic:'OCSVM', function:'Anomalia', synergies:['Normalization'], incompatibilities:[], desc:'One-Class SVM' },
  // Séries Temporais
  'ARIMA': { type:'model', inputs:['timeseries_data'], outputs:['forecast'], logic:'ARIMA', function:'Previsão', synergies:['Time Series Analysis'], incompatibilities:[], desc:'ARIMA' },
  'SARIMA': { type:'model', inputs:['timeseries_data'], outputs:['seasonal_forecast'], logic:'SARIMA', function:'Previsão', synergies:['Time Series Analysis'], incompatibilities:[], desc:'SARIMA' },
  'Prophet': { type:'model', inputs:['timeseries_data'], outputs:['prophet_forecast'], logic:'Prophet', function:'Previsão', synergies:['Time Series Analysis'], incompatibilities:[], desc:'Facebook Prophet' },
  'Exponential Smoothing': { type:'model', inputs:['timeseries_data'], outputs:['smooth_forecast'], logic:'ExpSmooth', function:'Previsão', synergies:['Time Series Analysis'], incompatibilities:[], desc:'Holt-Winters' },
  // NLP
  'Sentiment Analysis': { type:'model', inputs:['text_data'], outputs:['sentiment'], logic:'Sentiment', function:'NLP', synergies:['Text Processing'], incompatibilities:[], desc:'Análise de Sentimento' },
  'Named Entity Recognition': { type:'model', inputs:['text_data'], outputs:['entities'], logic:'NER', function:'NLP', synergies:['Tokenization'], incompatibilities:[], desc:'NER' },
  'Text Classification': { type:'model', inputs:['tfidf_vectors'], outputs:['text_class'], logic:'TextClass', function:'NLP', synergies:['TF-IDF'], incompatibilities:[], desc:'Classificação de Texto' },
  // Ensemble
  'Voting Classifier': { type:'model', inputs:['predictions_list'], outputs:['voted'], logic:'Voting', function:'Ensemble', synergies:['Ensemble Methods'], incompatibilities:[], desc:'Voting' },
  'Stacking': { type:'model', inputs:['predictions_list'], outputs:['stacked'], logic:'Stacking', function:'Ensemble', synergies:['Ensemble Methods'], incompatibilities:[], desc:'Stacking' },
  'Bagging': { type:'model', inputs:['clean_data'], outputs:['bagging_pred'], logic:'Bagging', function:'Ensemble', synergies:['Random Forest'], incompatibilities:[], desc:'Bagging' },
  'AdaBoost': { type:'model', inputs:['clean_data'], outputs:['adaboost_pred'], logic:'AdaBoost', function:'Boosting', synergies:['Decision Tree'], incompatibilities:[], desc:'AdaBoost' },
  'JSON Export': { type:'output', inputs:['predictions_list'], outputs:[], logic:'JSON', function:'Exportação', synergies:[], incompatibilities:[], desc:'Exporta JSON' },
  'CSV Export': { type:'output', inputs:['predictions_list'], outputs:[], logic:'CSV', function:'Exportação', synergies:[], incompatibilities:[], desc:'Exporta CSV' },
  'Parquet Export': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Parquet', function:'Exportação', synergies:[], incompatibilities:[], desc:'Exporta Parquet' },
  'Excel Export': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Excel', function:'Exportação', synergies:[], incompatibilities:[], desc:'Exporta Excel' },
  'Database Save': { type:'output', inputs:['predictions_list'], outputs:[], logic:'DB', function:'Armazenamento', synergies:[], incompatibilities:[], desc:'Salva no Banco' },
  'Visualization': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Viz', function:'Visualização', synergies:[], incompatibilities:[], desc:'Gráficos interativos' },
  'Dashboard': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Dashboard', function:'Visualização', synergies:[], incompatibilities:[], desc:'Dashboard' },
  'Report Generation': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Report', function:'Relatório', synergies:[], incompatibilities:[], desc:'Gera relatório' },
  'PDF Report': { type:'output', inputs:['predictions_list'], outputs:[], logic:'PDF', function:'Relatório', synergies:[], incompatibilities:[], desc:'Relatório em PDF' },
  'Email Alert': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Email', function:'Notificação', synergies:[], incompatibilities:[], desc:'Alerta por e-mail' },
  'Slack Notification': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Slack', function:'Notificação', synergies:[], incompatibilities:[], desc:'Notificação Slack' },
  'API Endpoint': { type:'output', inputs:['predictions_list'], outputs:[], logic:'API', function:'Serviço', synergies:[], incompatibilities:[], desc:'Disponibiliza API' },
  'Model Registry': { type:'output', inputs:['model_object'], outputs:[], logic:'Registry', function:'Armazenamento', synergies:[], incompatibilities:[], desc:'Registra modelo' },
  'Real-time Stream': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Stream', function:'Streaming', synergies:[], incompatibilities:[], desc:'Saída em streaming' },
  'Cloud Storage': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Cloud', function:'Armazenamento', synergies:[], incompatibilities:[], desc:'Salva na nuvem' },
  'Metrics Logger': { type:'output', inputs:['metrics'], outputs:[], logic:'Log', function:'Monitoramento', synergies:[], incompatibilities:[], desc:'Log de métricas' },
  'Threshold Trigger': { type:'logic', inputs:['numerical_value'], outputs:['signal'], logic:'Greater/Less', function:'Gatilho', synergies:['Expected Value','Kelly Criterion'], incompatibilities:[], desc:'Dispara sinal se valor > limite (ex: Odd > 2.0)' },
  'Logic Gate (AND/OR)': { type:'logic', inputs:['signal_a', 'signal_b'], outputs:['final_signal'], logic:'Boolean', function:'Filtro', synergies:[], incompatibilities:[], desc:'Combina condições (ex: IA deu OK AND Volume é alto)' },
  'Filter Gate': { type:'logic', inputs:['any_data'], outputs:['filtered_data'], logic:'Filter', function:'Filtro', synergies:[], incompatibilities:[], desc:'Bloqueia dados que não atendem requisitos mínimos' },
  'Cooldown Timer': { type:'logic', inputs:['signal'], outputs:['safe_signal'], logic:'Timer', function:'Segurança', synergies:[], incompatibilities:[], desc:'Evita disparos múltiplos seguidos (Proteção de banca)' },
  'Kill Switch': { type:'logic', inputs:['metrics'], outputs:['stop_signal'], logic:'EmergencyStop', function:'Segurança', synergies:['Data Drift Detector'], incompatibilities:[], desc:'Para o sniper se o prejuízo atingir um limite' },
  'Confidence Buffer': { type:'logic', inputs:['predictions_list'], outputs:['stable_prediction'], logic:'Median/Mean', function:'Estabilização', synergies:[], incompatibilities:[], desc:'Espera por 3 sinais iguais antes de disparar' },
  'SHAP Explainer': { type:'model', inputs:['model_object', 'tabular_data'], outputs:['importance_score'], logic:'SHAP', function:'Explicabilidade', synergies:['Visualization'], incompatibilities:[], desc:'Mostra quais variáveis causaram o Snipe' },
  'LLM Sentiment Analysis': { type:'model', inputs:['text_data'], outputs:['sentiment_score'], logic:'GPT/Claude', function:'NLP Pro', synergies:['Twitter Scraping','Web Scraping'], incompatibilities:[], desc:'Analisa notícias em tempo real para score de confiança' },
  'Market Regime Classifier': { type:'model', inputs:['timeseries_data'], outputs:['market_state'], logic:'HiddenMarkov', function:'Contexto', synergies:['Volatility Analysis'], incompatibilities:[], desc:'Detecta se o mercado está Tendência ou Lateral' },
  'Pattern Recognition': { type:'model', inputs:['image_data'], outputs:['pattern_found'], logic:'CNN_Custom', function:'Visão', synergies:['Gráficos'], incompatibilities:[], desc:'Detecta padrões visuais em gráficos' },
  'Data Drift Detector': { type:'output', inputs:['incoming_data'], outputs:['drift_alert'], logic:'Statistical_Test', function:'Monitoramento', synergies:[], incompatibilities:[], desc:'Avisa se o mercado mudou drasticamente (Modelo ficou obsoleto)' },
  'Real-time PnL Tracker': { type:'output', inputs:['results_data'], outputs:['profit_loss_stats'], logic:'Financial', function:'Financeiro', synergies:['Dashboard'], incompatibilities:[], desc:'Calcula Lucro/Prejuízo real vs Esperado' },
  'Backtest Simulator': { type:'model', inputs:['historical_data', 'logic_gate'], outputs:['simulation_report'], logic:'Backtest_Engine', function:'Validação', synergies:['Monte Carlo Simulation'], incompatibilities:[], desc:'Simula a estratégia no passado antes de ligar o real' },
  'Accuracy Monitor': { type:'output', inputs:['predictions', 'real_outcomes'], outputs:['precision_metrics'], logic:'Confusion_Matrix', function:'Métricas', synergies:['Email Alert'], incompatibilities:[], desc:'Mede a taxa de acerto das últimas X entradas' },
  'Odds/Prob Converter': { type:'preprocessing', inputs:['odds_decimal'], outputs:['implied_probability'], logic:'Math_Convert', function:'Conversão', synergies:['Kelly Criterion'], incompatibilities:[], desc:'Converte Odd para Probabilidade Real e vice-versa' },
  'Hedge Calculator': { type:'model', inputs:['active_bet', 'new_odds'], outputs:['hedge_amount'], logic:'Hedge_Formula', function:'Proteção', synergies:[], incompatibilities:[], desc:'Calcula quanto apostar contra para garantir lucro' },
  'Time Window Filter': { type:'logic', inputs:['timestamp'], outputs:['is_active'], logic:'Schedule', function:'Filtro', synergies:[], incompatibilities:[], desc:'Permite operação apenas em horários específicos' },
  'Data Normalizer (Z-Score)': { type:'preprocessing', inputs:['raw_numbers'], outputs:['normalized_numbers'], logic:'ZScore', function:'Limpeza', synergies:['Neural Network'], incompatibilities:[], desc:'Coloca dados diferentes na mesma escala' },
  'Webhook Sniper': { type:'output', inputs:['final_signal'], outputs:[], logic:'HTTP_POST', function:'Execução', synergies:['Telegram Bot'], incompatibilities:[], desc:'Envia ordem de compra/aposta para API externa' },
  'Sound Alert': { type:'output', inputs:['signal'], outputs:[], logic:'Audio', function:'Notificação', synergies:[], incompatibilities:[], desc:'Toca um alarme sonoro quando houver oportunidade' },
  'Discord Webhook': { type:'output', inputs:['predictions_list'], outputs:[], logic:'DiscordAPI', function:'Notificação', synergies:[], incompatibilities:[], desc:'Envia card detalhado para canal do Discord' },
'Poisson Distribution': { type:'preprocessing', inputs:['tabular_data'], outputs:['prob_distribution'], logic:'Poisson', function:'Estatística', synergies:['Sports Prediction'], incompatibilities:[], desc:'Cálculo de probabilidade de eventos raros' },
'Monte Carlo Simulation': { type:'model', inputs:['normalized_data'], outputs:['simulation_results'], logic:'MonteCarlo', function:'Simulação', synergies:['Risk Analysis'], incompatibilities:[], desc:'Simula milhares de cenários' },
'Bayesian Inference': { type:'model', inputs:['tabular_data'], outputs:['posterior_prob'], logic:'Bayesian', function:'Probabilidade', synergies:['Online Learning'], incompatibilities:[], desc:'Atualiza probabilidade com novos dados' },
'Elo Rating System': { type:'preprocessing', inputs:['tabular_data'], outputs:['rankings'], logic:'Elo', function:'Ranking', synergies:['Matchmaking'], incompatibilities:[], desc:'Sistema de classificação de força relativa' },
'Kelly Criterion': { type:'model', inputs:['win_probability', 'odds'], outputs:['bet_size'], logic:'Kelly', function:'Gestão de Banca', synergies:['Expected Value'], incompatibilities:[], desc:'Calcula o tamanho ideal da aposta' },
'Expected Value (EV)': { type:'model', inputs:['win_probability', 'odds'], outputs:['ev_score'], logic:'EV', function:'Decisão', synergies:['Profitability Analysis'], incompatibilities:[], desc:'Calcula o valor esperado (+EV/-EV)' },
'Arbitrage Detector': { type:'model', inputs:['multi_source_odds'], outputs:['arb_opportunity'], logic:'Arb', function:'Detecção', synergies:['Real-time Stream'], incompatibilities:[], desc:'Detecta arbitragem entre fontes' },
'Drawdown Analysis': { type:'output', inputs:['pnl_history'], outputs:['risk_metrics'], logic:'Drawdown', function:'Risco', synergies:['Backtesting'], incompatibilities:[], desc:'Calcula queda máxima de capital' },
'Cross-Validation': { type:'preprocessing', inputs:['tabular_data'], outputs:['cv_folds'], logic:'KFold', function:'Validação', synergies:['Grid Search'], incompatibilities:[], desc:'Validação cruzada K-Fold' },
'Confusion Matrix': { type:'output', inputs:['classification'], outputs:['matrix_plot'], logic:'ConfusionMatrix', function:'Métricas', synergies:['Visualization'], incompatibilities:[], desc:'Matriz de confusão' },
'ROC/AUC Curve': { type:'output', inputs:['classification'], outputs:['roc_plot'], logic:'ROC', function:'Métricas', synergies:['Binary Classification'], incompatibilities:[], desc:'Curva ROC e área sob a curva' },
'Backtesting Engine': { type:'model', inputs:['historical_data', 'strategy'], outputs:['performance_report'], logic:'Backtest', function:'Validação', synergies:['Financial Models'], incompatibilities:[], desc:'Testa estratégia em dados históricos' },
'Telegram Bot': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Telegram', function:'Notificação', synergies:[], incompatibilities:[], desc:'Envia alertas via Telegram' },
'Webhook Trigger': { type:'output', inputs:['predictions_list'], outputs:[], logic:'Webhook', function:'Integração', synergies:[], incompatibilities:[], desc:'Envia dados para URL externa' },
'Auto-Stop Loss': { type:'model', inputs:['realtime_data'], outputs:['exit_signal'], logic:'StopLoss', function:'Proteção', synergies:['Risk Management'], incompatibilities:[], desc:'Gera sinal de saída por perda' },
'Grid Search Tuner': { type:'preprocessing', inputs:['model_object'], outputs:['best_params'], logic:'GridSearch', function:'Otimização', synergies:['Hyperparameter Tuning'], incompatibilities:[], desc:'Busca exaustiva de parâmetros' },
'Odds API': { type:'data', inputs:[], outputs:['multi_source_odds'], logic:'API', function:'Coleta', synergies:['Arbitrage Detector'], incompatibilities:[], desc:'Coleta odds de múltiplas casas' },
'Twitter Sentiment': { type:'data', inputs:[], outputs:['text_data'], logic:'Social', function:'Sentimento', synergies:['NLP'], incompatibilities:[], desc:'Sentimento das redes sociais' },
'Market Depth (L2)': { type:'data', inputs:[], outputs:['orderbook_data'], logic:'L2', function:'Trading', synergies:['Volume Analysis'], incompatibilities:[], desc:'Dados de profundidade de mercado' },
'Moving Average (SMA/EMA)': { type:'preprocessing', inputs:['timeseries_data'], outputs:['smooth_signal'], logic:'MA', function:'Filtro', synergies:['Trend Following'], incompatibilities:[], desc:'Médias móveis simples e exponenciais' },
'RSI Indicator': { type:'preprocessing', inputs:['timeseries_data'], outputs:['rsi_values'], logic:'RSI', function:'Momento', synergies:['Overbought/Oversold'], incompatibilities:[], desc:'Índice de Força Relativa' },
'Bollinger Bands': { type:'preprocessing', inputs:['timeseries_data'], outputs:['bands'], logic:'Bollinger', function:'Volatilidade', synergies:['Reversion'], incompatibilities:[], desc:'Bandas de volatilidade' },
'Auto-Model Selector': { 
  type:'logic', 
  inputs:['tabular_data', 'target_variable'], 
  outputs:['best_model_prediction', 'winning_model_name'], 
  logic:'Tournament', 
  function:'Otimização', 
  desc:'Roda vários modelos em paralelo e escolhe o que tem maior precisão no momento.' 
},'Data Validator': { 
  type:'preprocessing', 
  inputs:['tabular_data'], 
  outputs:['validated_data', 'error_report'], 
  logic:'Schema_Check', 
  function:'Qualidade', 
  desc:'Verifica se os dados do CSV não estão corrompidos ou com valores impossíveis (ex: Odd negativa).' 
},'Quick Backtester': { 
  type:'logic', 
  inputs:['model_prediction', 'historical_results'], 
  outputs:['accuracy_score'], 
  logic:'Backtest', 
  function:'Validação', 
  desc:'Antes de você apostar, ele testa a Polynomial Regression nos últimos 10 minutos do CSV para ver se ela teria acertado.' 
},'Data Sanitizer': { 
    type:'preprocessing', 
    inputs:['tabular_data'], 
    outputs:['clean_data'], 
    logic:'Outlier_Remover', 
    function:'Limpeza', 
    synergies:['Normalization'], 
    desc:'Remove linhas com erros ou valores absurdos que estragam a média do modelo.' 
  },

  'Cross-Check API': { 
    type:'data', 
    inputs:['csv_prediction'], 
    outputs:['is_valid'], 
    logic:'API_Compare', 
    function:'Validação', 
    synergies:['CSV Input'], 
    desc:'Cruza o que o CSV previu com o que está acontecendo AGORA no mercado real.' 
  },

  'Execution Trigger': { 
    type:'output', 
    inputs:['monte_carlo_ok', 'model_signal'], 
    outputs:['TRADE_EXECUTION'], 
    logic:'Binary_Gate', 
    function:'Ação', 
    synergies:['Webhook Sniper'], 
    desc:'O GATILHO: Só atira se o Modelo deu sinal E o Monte Carlo aprovou o risco.' 
  },
  'Value at Risk (VaR)': { 
    type:'model', 
    inputs:['profit_loss_history'], 
    outputs:['max_expected_loss'], 
    logic:'Statistical_Percentile', 
    function:'Risco', 
    synergies:['Monte Carlo Simulation', 'Drawdown Analyzer'], 
    incompatibilities:[], 
    desc:'Calcula a perda máxima esperada com 95% de confiança.  Use com Monte Carlo para estressar os piores cenários.' 
  },

  'Risk of Ruin': { 
    type:'model', 
    inputs:['win_rate', 'avg_odds', 'bankroll'], 
    outputs:['ruin_probability'], 
    logic:'Statistical_Formula', 
    function:'Proteção', 
    synergies:['Kelly Criterion', 'Quick Backtester'], 
    incompatibilities:[], 
    desc:'Probabilidade matemática de quebrar a banca.  O Quick Backtester fornece a taxa de acerto real para este cálculo.' 
  },

  'Drawdown Analyzer': { 
    type:'output', 
    inputs:['equity_curve'], 
    outputs:['max_drawdown_stats'], 
    logic:'Peak_to_Valley', 
    function:'Métricas', 
    synergies:['Backtest Engine', 'Sharpe Ratio'], 
    incompatibilities:[], 
    desc:'Mede a maior queda de capital histórica.Essencial para calcular o Sharpe Ratio e entender a volatilidade do lucro.' 
  },

  'Sharpe Ratio': { 
    type:'output', 
    inputs:['returns', 'risk_free_rate'], 
    outputs:['efficiency_score'], 
    logic:'Risk_Adjusted_Return', 
    function:'Métricas', 
    synergies:['Expected Value (EV+)', 'Optimization'], 
    incompatibilities:[], 
    desc:'Indica se o lucro compensa o risco.  Ajuda o Auto-Model Selector a escolher modelos que não sejam apenas lucrativos, mas estáveis.' 
  },

  'Kelly Criterion': { 
    type:'model', 
    inputs:['win_prob', 'odds'], 
    outputs:['bet_percentage'], 
    logic:'Kelly_Formula', 
    function:'Gestão de Banca',
    synergies:['Expected Value (EV+)', 'Risk of Ruin'], 
    desc:'Calcula o tamanho ideal da aposta.  Use o EV+ para confirmar se há vantagem antes de calcular o valor.' 
  },

  'Expected Value (EV+)': { 
    type:'model', 
    inputs:['model_prob', 'market_odds'], 
    outputs:['ev_score'], 
    logic:'EV_Calculation', 
    function:'Decisão',
    synergies:['Implied Probability', 'Auto-Model Selector'], 
    desc:'Detecta apostas com valor positivo.  Precisa da Implied Probability para remover a taxa da casa e ver a odd real.' 
  },

  // ========== MÓDULOS DE PROBABILIDADE (probability) ==========
  'Poisson Distribution': { 
    type:'model', 
    inputs:['average_events'], 
    outputs:['probability_matrix'], 
    logic:'Poisson_Math', 
    function:'Probabilidade', 
    synergies:['Time-Decay Weighting', 'Normalization'], 
    incompatibilities:[], 
    desc:'Prevê chance de eventos exatos (gols/pontos).  O Time-Decay garante que a média de eventos seja baseada no histórico recente.' 
  },

  'Bayesian Updater': { 
    type:'model', 
    inputs:['prior_prob', 'live_data'], 
    outputs:['updated_prob'], 
    logic:'Bayes_Theorem', 
    function:'Probabilidade', 
    synergies:['Live API Connector', 'Poisson Distribution'], 
    incompatibilities:[], 
    desc:'Atualiza probabilidades em tempo real.  Usa o Poisson como "chute inicial" e a Live API para ajustar a crença com o jogo rolando.' 
  },

  'Implied Probability': { 
    type:'preprocessing', 
    inputs:['odds_data'], 
    outputs:['true_probability'], 
    logic:'Margin_Removal', 
    function:'Cálculo', 
    synergies:['Expected Value (EV+)', 'Outlier Sanitizer'], 
    incompatibilities:[], 
    desc:'Encontra a probabilidade real por trás das odds.  O Outlier Sanitizer deve limpar odds erradas antes deste cálculo.' 
  },

  'Standard Deviation (Z-Score)': { 
    type:'preprocessing', 
    inputs:['numerical_data'], 
    outputs:['outlier_signals'], 
    logic:'Gaussian_Stats', 
    function:'Estatística', 
    synergies:['Outlier Sanitizer', 'Normalization'], 
    incompatibilities:[], 
    desc:'Identifica dados anormais (Odds muito altas/baixas).  Ajuda a Normalization a não ser distorcida por valores extremos.' 
  },

  // ========== PROCESSAMENTOS AVANÇADOS (processing) ==========
  'Time-Decay Weighting': { 
    type:'preprocessing', 
    inputs:['historical_data'], 
    outputs:['weighted_data'], 
    logic:'Exponential_Decay', 
    function:'Tratamento', 
    synergies:['Polynomial Regression', 'Poisson Distribution'], 
    incompatibilities:[], 
    desc:'Dá peso maior a dados recentes em mercados voláteis.' 
  },

  'SMOTE (Imbalance Fix)': { 
    type:'preprocessing', 
    inputs:['imbalanced_data'], 
    outputs:['balanced_data'], 
    logic:'Synthetic_Oversampling', 
    function:'Tratamento', 
    synergies:['XGBoost', 'Random Forest'], 
    incompatibilities:[], 
    desc:'Equilibra dados, perforam melhor com classes equilibradas.' 
  },

  'Moving Average Convergence (MACD)': { 
    type:'preprocessing', 
    inputs:['timeseries_data'], 
    outputs:['momentum_signal'], 
    logic:'Dual_EMA', 
    function:'Sinal', 
    synergies:['Trend Following', 'Live API Connector'], 
    incompatibilities:[], 
    desc:'Detecta mudança de tendência de API para perceber quando o mercado está virando contra você.' 
  },

  'Outlier Sanitizer': { 
    type:'preprocessing', 
    inputs:['raw_data'], 
    outputs:['clean_data'], 
    logic:'Z-Score_Filter', 
    function:'Limpeza',
    synergies:['Standard Deviation (Z-Score)', 'Normalization'], 
    desc:'Remove lixo e erros de API e trabalhe apenas com dados reais.' 
  },

  'Feature Crosser': { 
    type:'preprocessing', 
    inputs:['feature_a', 'feature_b'], 
    outputs:['combined_feature'], 
    logic:'Interaction_Logic', 
    function:'Feature Engineering',
    synergies:['Polynomial Regression', 'XGBoost'], 
    desc:'Cria novas variáveis cruzadas as interações que ela precisa para criar curvas.' 
  },
'Risk Analysis': { 
    type:'model', 
    inputs:['ruin_probability', 'max_expected_loss', 'drawdown_stats'], 
    outputs:['risk_score', 'safety_verdict'], 
    logic:'Weighted_Risk_Assessment', 
    function:'Decisão Final', 
    synergies:['Risk of Ruin',"Monte Carlo", 'Value at Risk (VaR)', 'Drawdown Analyzer'], 
    incompatibilities:[], 
    desc:'O Cérebro do Risco: Cruza todos os indicadores.' 
  },

  'Sensitivity Analysis': { 
    type:'model', 
    inputs:['model_prediction', 'parameter_variants'], 
    outputs:['stability_report'], 
    logic:'What-If_Scenarios', 
    function:'Estresse', 
    synergies:['Polynomial Regression', 'Monte Carlo Simulation'], 
    incompatibilities:[], 
    desc:'Testa o que acontece se os dados mudarem 1% (ex: se a odd cair um pouco, o lucro ainda vale o risco?).' 
  },

  'Exposure Manager': { 
    type:'model', 
    inputs:['active_trades', 'bankroll'], 
    outputs:['exposure_limit'], 
    logic:'Capital_Allocation', 
    function:'Gestão', 
    synergies:['Kelly Criterion'], 
    incompatibilities:[], 
    desc:'Controla o valor total exposto no mercado. Impede que o Sniper atire se você já tiver muito dinheiro em risco ao mesmo tempo.' 
  },

  'Stress Tester': { 
    type:'model', 
    inputs:['historical_extremes'], 
    outputs:['survival_status'], 
    logic:'Worst_Case_Scenario', 
    function:'Segurança', 
    synergies:['Backtest Engine'], 
    incompatibilities:[], 
    desc:'Simula como a sua estratégia teria sobrevivido aos piores dias da história do mercado (ex: uma zebra histórica).' 
  },

  'Profit Objective (Take Profit)': { 
    type:'logic', 
    inputs:['current_profit'], 
    outputs:['stop_signal'], 
    logic:'Target_Reached', 
    function:'Meta', 
    synergies:['Real-time PnL Tracker'], 
    incompatibilities:[], 
    desc:'Define uma meta de lucro para o dia. Quando batida, o sniper "guarda o rifle" e para de operar para proteger o ganho.' 
  },
'SQLite Input': { 
    type:'data', inputs:[], outputs:['sql_data'], logic:'SQLite', function:'Leitura', 
    synergies:['Data Cleaning', 'Feature Selection'], incompatibilities:[], 
    desc:'Lê banco de dados local (rápido e leve para histórico).' 
  },
  'Excel Pro Input': { 
    type:'data', inputs:[], outputs:['spreadsheet_data'], logic:'XLSX', function:'Leitura', 
    synergies:['Normalization'], incompatibilities:[], 
    desc:'Importa abas específicas de arquivos Excel (.xlsx).' 
  },
  'Google Sheets Input': { 
    type:'data', inputs:[], outputs:['cloud_tabular'], logic:'GSheets_API', function:'Cloud', 
    synergies:['Manual Input'], incompatibilities:[], 
    desc:'Conecta direto em uma planilha do Google Drive (útil para gerir banca pelo celular).' 
  },
  'PostgreSQL/MySQL': { 
    type:'data', inputs:[], outputs:['sql_data'], logic:'SQL_Query', function:'Database', 
    synergies:['Data Sanitizer'], incompatibilities:[], 
    desc:'Conecta em bancos de dados relacionais robustos.' 
  },
  'Redis Cache Input': { 
    type:'data', inputs:[], outputs:['fast_data'], logic:'NoSQL_Cache', function:'Ultra-Fast', 
    synergies:['Real-time Stream'], incompatibilities:[], 
    desc:'Lê dados da memória RAM (milissegundos) para snipes que exigem velocidade extrema.' 
  },

  // ========== ENTRADAS EM TEMPO REAL (live) ==========
  'WebSocket Stream': { 
    type:'data', inputs:[], outputs:['live_stream'], logic:'WS_Socket', function:'Streaming', 
    synergies:['Bayesian Updater', 'Live API Connector'], incompatibilities:[], 
    desc:'Recebe dados que "empurram" (push) para o sniper sem precisar atualizar a página.' 
  },
  'Web Scraper (HTML)': { 
    type:'data', inputs:[], outputs:['scraped_text'], logic:'Puppeteer/BS4', function:'Coleta', 
    synergies:['NLP Sentiment Analysis', 'Data Cleaning'], incompatibilities:[], 
    desc:'Extrai dados direto do site (DOM) quando não há API disponível.' 
  },
  'Twitter/X Feed': { 
    type:'data', inputs:[], outputs:['social_text'], logic:'Social_API', function:'Sentimento', 
    synergies:['LLM Sentiment Analysis'], incompatibilities:[], 
    desc:'Captura tweets em tempo real sobre um time ou mercado.' 
  },
  'Odds API Connector': { 
    type:'data', inputs:[], outputs:['market_odds'], logic:'JSON_REST', function:'Market Data', 
    synergies:['Expected Value (EV+)', 'Arbitrage Detector'], incompatibilities:[], 
    desc:'Entrada especializada para cotações de múltiplas casas de apostas.' 
  },

  // ========== ENTRADAS DE CONFIGURAÇÃO E ESTADO (state) ==========
  'User UI Input': { 
    type:'data', inputs:[], outputs:['user_params'], logic:'GUI_Form', function:'Configuração', 
    synergies:['Kelly Criterion'], incompatibilities:[], 
    desc:'Caixa de entrada para o usuário digitar manualmente a banca atual ou risco desejado.' 
  },
  'Clipboard Input': { 
    type:'data', inputs:[], outputs:['raw_text'], logic:'Paste', function:'Utilitário', 
    synergies:['Data Parsing'], incompatibilities:[], 
    desc:'Pega dados que você copiou (Ctrl+C) e joga para dentro do Sniper.' 
  },
  'Historical Replay': { 
    type:'data', inputs:['historical_file'], outputs:['simulated_stream'], logic:'Time_Machine', function:'Debug', 
    synergies:['Quick Backtester'], incompatibilities:[], 
    desc:'Simula um fluxo em tempo real usando dados do passado (para ver o sniper trabalhando).' 
  },
'Data Parsing': {
  type: 'preprocessing',
  inputs: ['json_data', 'xml_data', 'api_data', 'scraped_data', 'text_data'],
  outputs: ['structured_data', 'tabular_data'],
  logic: 'Parser',
  function: 'Transformação',
  synergies: ['Data Cleaning', 'Feature Engineering', 'Normalization'],
  incompatibilities: [],
  desc: 'Converte dados semi-estruturados (JSON, XML, texto) em formato tabular ou estruturado'
},
'Data Schema Validator': { 
    type:'logic', 
    inputs:['tabular_data'], 
    outputs:['validated_data', 'error_log'], 
    logic:'Schema_Check', 
    function:'Validação', 
    synergies:['Data Cleaning', 'CSV Input'], 
    desc:'Garante que o CSV tem as colunas certas. Se vier uma coluna errada, o monstro para antes de dar erro.' 
  },

  'Model Serializer (Save/Load)': { 
    type:'output', 
    inputs:['model_object'], 
    outputs:['saved_file'], 
    logic:'Pickle/Joblib', 
    function:'Armazenamento', 
    synergies:['Auto-Model Selector'], 
    desc:'Salva o "treinamento". Assim você não precisa treinar tudo de novo toda vez que ligar.' 
  },

  'API Pusher (Webhook)': { 
    type:'output', 
    inputs:['json_data'], 
    outputs:['http_response'], 
    logic:'HTTP_POST', 
    function:'Integração', 
    synergies:['JSON Export'], 
    desc:'Pega o seu JSON Export e envia direto para o seu bot do Telegram ou para a corretora.' 
  },

  'Parallel Processor': { 
    type:'logic', 
    inputs:['tabular_data'], 
    outputs:['multi_threaded_data'], 
    logic:'Multi_Threading', 
    function:'Performance', 
    synergies:['Auto-Model Selector', 'Monte Carlo Simulation'], 
    desc:'Faz o monstro usar todos os núcleos do seu PC para rodar os 55 modelos ao mesmo tempo.' 
  },  'Boolean Gate (AND/OR/NOT)': { 
    type:'logic', inputs:['signal_a', 'signal_b'], outputs:['final_signal'], 
    logic:'Boolean_Logic', function:'Filtro', 
    synergies:['Threshold Trigger'], incompatibilities:[], 
    desc:'Combina condições. Ex: Só atira se (Modelo A deu OK) AND (Kelly Criterion > 2%).' 
  },
  'Value Comparator': { 
    type:'logic', inputs:['val_a', 'val_b'], outputs:['is_greater', 'is_less', 'is_equal'], 
    logic:'Comparison', function:'Filtro', 
    synergies:['Expected Value (EV+)'], incompatibilities:[], 
    desc:'Compara dois valores. Útil para ver se a Odd da Casa A é maior que a da Casa B.' 
  },
  'Switch / Case': { 
    type:'logic', inputs:['input_val'], outputs:['path_1', 'path_2', 'path_3'], 
    logic:'Branching', function:'Roteamento', 
    synergies:['Auto-Model Selector'], incompatibilities:[], 
    desc:'Desvia o fluxo de dados para caminhos diferentes dependendo do resultado do modelo.' 
  },
  'Signal Debounce (Cooldown)': { 
    type:'logic', inputs:['trigger'], outputs:['filtered_trigger'], 
    logic:'Timer_Delay', function:'Segurança', 
    synergies:['Execution Trigger'], incompatibilities:[], 
    desc:'Evita múltiplos disparos seguidos. Se o sniper atirou agora, ele espera X segundos para o próximo.' 
  },
  'Latch / Memory': { 
    type:'logic', inputs:['set', 'reset'], outputs:['state'], 
    logic:'Flip-Flop', function:'Estado', 
    synergies:['Kill Switch'], incompatibilities:[], 
    desc:'Guarda uma decisão. Ex: "Travar Sniper" até que eu venha manualmente dar um Reset.' 
  },

  // ========== LÓGICA DE FILTRAGEM (filtering) ==========
  'Range Filter': { 
    type:'logic', inputs:['value'], outputs:['in_range', 'out_range'], 
    logic:'Boundaries', function:'Validação', 
    synergies:['Normalization'], incompatibilities:[], 
    desc:'Só deixa o sinal passar se o valor estiver entre X e Y (Ex: Odd entre 1.5 e 2.5).' 
  },
  'Delta Trigger (Change)': { 
    type:'logic', inputs:['current_val'], outputs:['change_signal'], 
    logic:'Difference_Check', function:'Gatilho', 
    synergies:['Live API Connector'], incompatibilities:[], 
    desc:'Dispara um sinal apenas se o valor mudar bruscamente (Ex: Odd caiu 0.50 em 10 segundos).' 
  },
  'Priority Selector': { 
    type:'logic', inputs:['signal_high', 'signal_low'], outputs:['selected_signal'], 
    logic:'Priority_Queue', function:'Decisão', 
    synergies:['Risk Analysis'], incompatibilities:[], 
    desc:'Se dois modelos derem sinal, ele escolhe o que você marcou como mais confiável.' 
  },

  // ========== LÓGICA DE TEMPO E AGENDAMENTO (time-logic) ==========
  'Time Window': { 
    type:'logic', inputs:['signal'], outputs:['allowed_signal'], 
    logic:'Scheduler', function:'Filtro Temporal', 
    synergies:['Live API Connector'], incompatibilities:[], 
    desc:'Só permite o "Snipe" em horários específicos (Ex: 5 minutos antes do jogo começar).' 
  },
  'Sequencer': { 
    type:'logic', inputs:['start_signal'], outputs:['step_1', 'step_2', 'step_3'], 
    logic:'Step_Execution', function:'Automação', 
    synergies:['Backtest Engine'], incompatibilities:[], 
    desc:'Executa tarefas em ordem: 1. Limpa, 2. Analisa, 3. Atira, 4. Exporta JSON.' 
  },

  // ========== LÓGICA DE SEGURANÇA E ERRO (safety-logic) ==========
  'Emergency Kill Switch': { 
    type:'logic', inputs:['error_signal', 'loss_limit'], outputs:['system_off'], 
    logic:'Hard_Stop', function:'Segurança', 
    synergies:['Risk Analysis', 'Real-time PnL Tracker'], incompatibilities:[], 
    desc:'Desliga todo o programa instantaneamente se algo sair do controle.' 
  },
  'Majority Voting': { 
    type:'logic', inputs:['signals_list'], outputs:['voted_signal'], 
    logic:'Consensus', function:'Decisão Coletiva', 
    synergies:['Auto-Model Selector'], incompatibilities:[], 
    desc:'O Sniper só atira se a MAIORIA dos seus modelos (ex: 3 de 5) concordarem.' 
  },
  'Fallback Logic': { 
    type:'logic', inputs:['primary_data', 'backup_data'], outputs:['final_data'], 
    logic:'Redundancy', function:'Estabilidade', 
    synergies:['CSV Input', 'Live API Connector'], incompatibilities:[], 
    desc:'Se a API em tempo real cair, ele muda automaticamente para os dados do CSV/Backup.' 
  },
// ========== GATILHOS DE TEMPO REAL (live-logic) ==========
  'Signal Threshold Gate': { 
    type:'logic', 
    inputs:['updated_prob'], 
    outputs:['filtered_signal'], 
    logic:'Value_Limit', 
    function:'Filtro', 
    synergies:['Bayesian Updater', 'API Pusher'], 
    desc:'SÓ deixa passar para o Webhook se a probabilidade atingir um valor específico (ex: acima de 70%).' 
  },

  'Latency Monitor': { 
    type:'output', 
    inputs:['stream_timestamp'], 
    outputs:['delay_ms'], 
    logic:'Time_Diff', 
    function:'Performance', 
    synergies:['WebSocket Stream'], 
    desc:'Mede o atraso entre o jogo real e o seu sniper. Se o atraso for > 2 segundos, ele trava o tiro por segurança.' 
  },

  'Deduplication Filter': { 
    type:'logic', 
    inputs:['signal'], 
    outputs:['unique_signal'], 
    logic:'ID_Check', 
    function:'Segurança', 
    synergies:['API Pusher'], 
    desc:'Garante que o sniper só atire UMA VEZ por oportunidade, mesmo que o WebSocket mande 10 atualizações seguidas.' 
  },

  'State Machine': { 
    type:'logic', 
    inputs:['live_event'], 
    outputs:['system_state'], 
    logic:'FSM', 
    function:'Controle', 
    synergies:['WebSocket Stream'], 
    desc:'Gerencia os estados: ESPERANDO, ANALISANDO, ATIROU, COOLDOWN. Evita que o sniper tente atirar enquanto a aposta anterior ainda está sendo processada.' 
  },
 'Live API Connector': { 
    type:'data', 
    inputs:['endpoint_url', 'api_auth'], 
    outputs:['api_payload', 'status_code'], 
    logic:'REST_JSON', 
    function:'Coleta Real-time', 
    synergies:['WebSocket Stream', 'Bayesian Updater', 'Data Schema Validator'], 
    incompatibilities:[], 
    desc:'Conecta em APIs REST para buscar dados estruturados sob demanda. SINERGIA: Complementa o WebSocket (que é rápido mas instável) fornecendo a "verdade absoluta" dos dados estruturados para o Bayesian Updater.' 
  },
'Expected Value (EV+)': { 
    type:'model', 
    inputs:['model_probability', 'market_odds'], 
    outputs:['is_profitable', 'edge_value'], 
    logic:'EV_Formula', 
    function:'Decisão', 
    synergies:['Poisson Distribution', 'Live API Connector'], 
    desc:'O Cérebro Financeiro: Compara a sua probabilidade (Poisson) com a Odd da API. Só libera o Webhook se houver lucro matemático (Edge).' 
  },

  'Poisson Matrix Merger': { 
    type:'logic', 
    inputs:['poisson_a', 'poisson_b'], 
    outputs:['final_odds_matrix'], 
    logic:'Matrix_Multiplication', 
    function:'Processamento', 
    synergies:['Poisson Distribution'], 
    desc:'Cruza os dados das duas Poissons para gerar as probabilidades de placar exato (ex: 1x0, 2x1, etc).' 
  },

  'Safety Execution Gate': { 
    type:'logic', 
    inputs:['is_profitable', 'risk_score'], 
    outputs:['webhook_trigger'], 
    logic:'Binary_AND', 
    function:'Gatilho Final', 
    synergies:['Expected Value (EV+)', 'Risk Analysis'], 
    desc:'O Dedo no Gatilho: Só deixa o API Pusher agir se o lucro for real E o risco estiver controlado.' 
  },
'Main Event Loop': { 
    type:'logic', 
    inputs:['start_signal', 'stop_signal'], 
    outputs:['tick'], 
    logic:'Event_Emitter', 
    function:'Coração do Sistema', 
    synergies:['WebSocket Stream', 'Latency Monitor'], 
    desc:'O motor que mantém o monstro vivo. Define quantos "tiros" por segundo o sistema pode processar (Tick Rate).' 
  },

  'Recursion Guard': { 
    type:'logic', 
    inputs:['loop_input'], 
    outputs:['safe_output'], 
    logic:'Stack_Limit', 
    function:'Segurança', 
    synergies:['Bayesian Updater'], 
    desc:'Impede que o cálculo entre em um loop infinito matemático que trave o programa (Stack Overflow).' 
  },

  'Feedback Loop (Learning)': { 
    type:'logic', 
    inputs:['snipe_result', 'previous_prediction'], 
    outputs:['model_adjustment'], 
    logic:'Reinforcement_Learning', 
    function:'Auto-Ajuste', 
    synergies:['Auto-Model Selector', 'Quick Backtester'], 
    desc:'O monstro aprende com o erro: Se ele errou o tiro, ele ajusta os pesos da Poisson automaticamente para o próximo tick.' 
  },

  'Heartbeat Monitor': { 
    type:'output', 
    inputs:['tick_signal'], 
    outputs:['health_status'], 
    logic:'Watchdog_Timer', 
    function:'Monitoramento', 
    synergies:['API Pusher'], 
    desc:'Se o loop travar por mais de 500ms, este bloco reinicia a conexão ou envia um alerta de emergência.' 
  },

  'Execution Quota': { 
    type:'logic', 
    inputs:['webhook_request'], 
    outputs:['allowed_request'], 
    logic:'Rate_Limiter', 
    function:'Gestão de Fluxo', 
    synergies:['API Pusher (Webhook)'], 
    desc:'Limita o número de ordens enviadas por minuto. Protege você de ser banido pela API por excesso de requisições.' 
  },
// ========== EXECUÇÃO FINAL (execution) ==========
  'API Pusher (Webhook)': { 
    type:'output', 
    inputs:['json_payload', 'execution_signal'], 
    outputs:['http_status', 'transaction_id'], 
    logic:'HTTP_POST_Async', 
    function:'Execução de Ordem', 
    synergies:['Execution Quota', 'JSON Export', 'Reinforcement Learning'], 
    incompatibilities:[], 
    desc:'O gatilho final. Envia a ordem para a corretora ou bot via POST. Ele retorna o ID da transação para que o sistema saiba que o "tiro" foi disparado com sucesso.' 
  },

 
  'Reinforcement Learning (RL)': { 
    type:'model', 
    inputs:['last_prediction', 'actual_result', 'reward_signal'], 
    outputs:['adjusted_weights', 'strategy_update'], 
    logic:'Q-Learning / Policy_Gradient', 
    function:'Aprendizado por Reforço', 
    synergies:['Auto-Model Selector', 'Quick Backtester', 'API Pusher (Webhook)'], 
    incompatibilities:[], 
    desc:'O cérebro evolutivo. Ele recebe o resultado do "tiro" (ganhou ou perdeu) e calcula a recompensa (Reward). Com base nisso, ele ajusta os pesos da Poisson e da Polynomial Regression para o próximo ciclo do loop.' 
  },

  'Reward Calculator': { 
    type:'logic', 
    inputs:['profit_loss', 'prediction_accuracy'], 
    outputs:['reward_signal'], 
    logic:'Score_Formula', 
    function:'Feedback', 
    synergies:['Reinforcement Learning', 'Real-time PnL Tracker'], 
    desc:'Transforma o lucro ou prejuízo em um sinal numérico para o Reinforcement Learning. (Ex: Lucro alto = +10 pontos, Prejuízo = -20 pontos).' 
  },
'Execution Quota': { 
    type:'logic', 
    inputs:['execution_request'], 
    outputs:['authorized_request', 'quota_exhausted_alert'], 
    logic:'Token_Bucket / Rate_Limiter', 
    function:'Gestão de Fluxo / Segurança', 
    synergies:['API Pusher (Webhook)', 'Main Event Loop', 'Risk Analysis'], 
    incompatibilities:[], 
    desc: 'O Controlador de Tráfego: Define limites rígidos de quantas ordens.' 
  },
'Event Loop': { 
    type:'logic', 
    inputs:['start_signal', 'stop_signal', 'tick_rate_hz'], 
    outputs:['tick_pulse', 'cycle_duration_ms', 'loop_count'], 
    logic:'Async_High_Resolution_Timer', 
    function:'Coração / Orquestração', 
    synergies:['WebSocket Stream', 'Latency Monitor', 'Execution Quota', 'Heartbeat Monitor'], 
    incompatibilities: [], 
    desc: 'O Motor Central: Dispara pulsos (ticks) em alta frequência para manter o monstro processando.' 
  },
 'System Mode Controller': { 
    type:'logic', 
    inputs:['mode_toggle'], // 0: Sandbox, 1: Live
    outputs:['active_path'], 
    logic:'Production_Bridge', 
    function:'Segurança de Ambiente', 
    synergies:['API Pusher (Webhook)', 'User UI Input'], 
    desc:'Alterna entre modo "Simulação" (apenas loga os resultados) e modo "Live" (envia dinheiro real). Essencial para testar novos blocos sem risco.' 
  },

  'Consensus Engine (The Judge)': { 
    type:'logic', 
    inputs:['model_signals_list'], 
    outputs:['unified_decision'], 
    logic:'Majority_Voting / Weighted_Average', 
    function:'Orquestração de Modelos', 
    synergies:['Auto-Model Selector', 'Safety Execution Gate'], 
    desc:'Gerencia a decisão dos 55 modelos. Ele pode exigir que, por exemplo, 70% dos modelos concordem antes de enviar o sinal para o Gatilho Final.' 
  },

  'Circuit Breaker (Anti-Panic)': { 
    type:'logic', 
    inputs:['error_rate', 'loss_streak'], 
    outputs:['system_lock'], 
    logic:'Fail-Safe_Interrupter', 
    function:'Proteção de Infraestrutura', 
    synergies:['Real-time PnL Tracker', 'Heartbeat Monitor'], 
    desc:'Se o sistema detectar 3 erros seguidos de API ou uma sequência de perdas impossível, ele corta a energia de todos os Webhooks instantaneamente.' 
  },

  'Pipeline Bypass': { 
    type:'logic', 
    inputs:['priority_signal', 'standard_data'], 
    outputs:['fast_path_data'], 
    logic:'Short_Circuit_Logic', 
    function:'Otimização de Latência', 
    synergies:['WebSocket Stream', 'Execution Quota'], 
    desc:'Se um sinal for ultra-confiável, este bloco pula etapas de processamento lentas (como NLP pesado) para garantir que o tiro chegue antes da odd mudar.' 
  },

  'Global Dependency Lock': { 
    type:'logic', 
    inputs:['block_status_array'], 
    outputs:['readiness_signal'], 
    logic:'Dependency_Graph_Check', 
    function:'Validação de Estado', 
    synergies:['Data Schema Validator', 'Event Loop'], 
    desc:'Verifica se todos os blocos críticos (API, Poisson, Risk).' 
  },

  'Hierarchy Master Switch': { 
    type:'logic', 
    inputs:['master_signal', 'safety_gate_signal'], 
    outputs:['final_authorization'], 
    logic:'Top_Down_Control', 
    function:'Autoridade Máxima', 
    synergies:['Safety Execution Gate', 'User UI Input'], 
    desc:'O último nível de controle. Mesmo que o "Safety Execution Gate" diga SIM.' 
  },
'Moving Average (SMA/EMA)': { 
    type:'logic', 
    inputs:['data_series', 'period'], 
    outputs:['smoothed_value'], 
    logic:'Rolling_Mean', 
    function:'Suavização', 
    synergies:['Trend Following', 'MACD'], 
    desc:'Fórmula: (Soma dos últimos N valores / N). Limpa o "ruído" dos dados para mostrar a tendência real sem oscilações bruscas.' 
  },

  'Z-Score (Standardization)': { 
    type:'logic', 
    inputs:['value', 'mean', 'std_dev'], 
    outputs:['z_score'], 
    logic:'(x - μ) / σ', 
    function:'Normalização Estatística', 
    synergies:['Anomaly Detection', 'Outlier Sanitizer'], 
    desc:'Mede quantos desvios padrões um dado está longe da média. Essencial para identificar se uma Odd ou preço está "anormal" (Oportunidade de Snipe).' 
  },

  'Expected Value (EV)': { 
    type:'logic', 
    inputs:['probability', 'odds'], 
    outputs:['ev_score'], 
    logic:'(P * O) - 1', 
    function:'Cálculo de Lucratividade', 
    synergies:['Kelly Criterion', 'Safety Execution Gate'], 
    desc:'A fórmula mestre: Multiplica sua chance de ganhar pela cotação. Se o resultado for > 0, você tem uma "Aposta de Valor" (Edge).' 
  },

  'Sigmoid Function': { 
    type:'logic', 
    inputs:['raw_score'], 
    outputs:['probability_0_1'], 
    logic:'1 / (1 + e^-x)', 
    function:'Conversão de Probabilidade', 
    synergies:['Neural Network', 'Logistic Regression'], 
    desc:'Esmaga qualquer número para um intervalo entre 0 e 1. Usada para transformar resultados de modelos complexos em uma porcentagem de confiança (0% a 100%).' 
  },

  'Percentage Change (Delta %)': { 
    type:'logic', 
    inputs:['old_value', 'new_value'], 
    outputs:['pct_diff'], 
    logic:'((new - old) / old) * 100', 
    function:'Medição de Impulso', 
    synergies:['Delta Trigger', 'Volatility Analysis'], 
    desc:'Calcula a variação percentual. Crucial para o Sniper detectar "pumps" ou quedas rápidas no mercado em tempo real.' 
  },

  'Pearson Correlation': { 
    type:'logic', 
    inputs:['series_a', 'series_b'], 
    outputs:['correlation_coeff'], 
    logic:'Cov(X,Y) / (σX * σY)', 
    function:'Análise de Relação', 
    synergies:['Feature Selection', 'Market Regime Classifier'], 
    desc:'Mede o quanto duas variáveis se movem juntas. Útil para saber se o desempenho de um time/ativo afeta diretamente o outro.' 
  },
  'Softmax (Multi-Class Prob)': { 
    type:'logic', 
    inputs:['raw_scores_list'], 
    outputs:['probabilities_0_1'], 
    logic:'exp(xi) / sum(exp(xj))', 
    function:'Normalização de IA', 
    synergies:['Neural Network', 'Consensus Engine'], 
    desc:'Pega os scores brutos de vários modelos e transforma em probabilidades que somam 100%. Essencial para prever resultados múltiplos (ex: Vitória, Empate, Derrota).' 
  },

  'ReLU (Rectified Linear Unit)': { 
    type:'logic', 
    inputs:['weighted_sum'], 
    outputs:['activated_output'], 
    logic:'max(0, x)', 
    function:'Ativação de Neurônio', 
    synergies:['Neural Network', 'Deep Learning'], 
    desc:'A fórmula mais usada em Redes Neurais modernas. Ela "ativa" o sinal apenas se ele for positivo, permitindo que a IA aprenda padrões complexos e ignore ruídos negativos.' 
  },

  'Cross-Entropy Loss': { 
    type:'logic', 
    inputs:['prediction', 'actual_result'], 
    outputs:['error_score'], 
    logic:'-sum(y * log(y_hat))', 
    function:'Cálculo de Erro', 
    synergies:['Reinforcement Learning', 'Reward Calculator'], 
    desc:'Mede a "surpresa" ou erro da IA. Quanto maior o erro, mais a IA precisa ajustar seus pesos. É o que diz para o monstro: "Você errou feio, mude sua estratégia".' 
  },

  'Argmax (Decision Picker)': { 
    type:'logic', 
    inputs:['probabilities_list'], 
    outputs:['chosen_index', 'confidence_level'], 
    logic:'index(max(list))', 
    function:'Tomada de Decisão', 
    synergies:['API Pusher (Webhook)', 'Safety Execution Gate'], 
    desc:'A lógica final da IA: olha para todas as probabilidades e escolhe a maior. Ex: Se Vitória=70% e Empate=30%, o Argmax escolhe a Vitória e envia o sinal.' 
  },

  'Cosine Similarity': { 
    type:'logic', 
    inputs:['vector_a', 'vector_b'], 
    outputs:['similarity_score'], 
    logic:'(A·B) / (||A||*||B||)', 
    function:'Reconhecimento de Padrões', 
    synergies:['Historical Replay', 'Feature Store'], 
    desc:'Mede o quão parecido o cenário de AGORA é com um cenário do PASSADO. Se a similaridade for > 0.9, o sniper sabe que esse padrão costuma resultar em vitória.' 
  },

  'Learning Rate Decay': { 
    type:'logic', 
    inputs:['initial_lr', 'epoch_count'], 
    outputs:['current_lr'], 
    logic:'lr / (1 + decay * epoch)', 
    function:'Otimização de Aprendizado', 
    synergies:['Reinforcement Learning', 'Event Loop'], 
    desc:'Diminui a velocidade de aprendizado conforme a IA fica mais inteligente. Isso evita que ela mude de ideia drasticamente após um único erro, mantendo a estabilidade do sniper.' 
  },
'Gaussian PDF (Normal Distribution)': { 
    type:'logic', 
    inputs:['value', 'mean', 'variance'], 
    outputs:['probability_density'], 
    logic:'1/√(2πσ²) * e^-((x-μ)²/2σ²)', 
    function:'Probabilidade Normal', 
    synergies:['Monte Carlo Simulation', 'Linear Regression'], 
    desc:'Calcula a curva de sino. SINERGIA: Alimenta a Monte Carlo Simulation com amostras realistas e ajuda a Linear Regression a entender a distribuição dos erros.' 
  },

  'Polynomial Basis Expansion': { 
    type:'logic', 
    inputs:['input_x', 'degree'], 
    outputs:['expanded_features'], 
    logic:'[x, x², x³, ..., xⁿ]', 
    function:'Lógica Não-Linear', 
    synergies:['Polynomial Regression', 'Neural Network'], 
    desc:'Transforma uma entrada linear em múltiplas dimensões curvas. SINERGIA: É o pré-requisito para a Polynomial Regression e ajuda a Neural Network a encontrar padrões não-lineares em dados simples.' 
  },

  'Linear Weighted Sum': { 
    type:'logic', 
    inputs:['features', 'weights', 'bias'], 
    outputs:['dot_product'], 
    logic:'Σ(wi * xi) + b', 
    function:'Lógica Linear', 
    synergies:['Linear Regression', 'Neural Network', 'Transformer'], 
    desc:'A base de toda a IA. SINERGIA: É a fórmula principal da Linear Regression e o primeiro passo dentro de cada neurônio de uma Neural Network ou Transformer.' 
  },

  'Logit/Sigmoid Link': { 
    type:'logic', 
    inputs:['linear_output'], 
    outputs:['probability_score'], 
    logic:'ln(p/(1-p)) / 1/(1+e^-z)', 
    function:'Lógica Não-Linear', 
    synergies:['Neural Network', 'Linear Regression'], 
    desc:'Faz a ponte entre um resultado numérico infinito e uma probabilidade de 0 a 1. SINERGIA: Transforma uma regressão linear em uma classificação probabilística para a IA.' 
  },

  'Scaled Dot-Product Attention': { 
    type:'logic', 
    inputs:['query', 'key', 'value'], 
    outputs:['attention_context'], 
    logic:'Softmax(QKᵀ / √dk)V', 
    function:'Lógica Sequencial Não-Linear', 
    synergies:['Transformer', 'Neural Network'], 
    desc:'A fórmula que revolucionou a IA. SINERGIA: É o coração do Transformer, permitindo que ele foque no que é importante dentro de uma série temporal de dados.' 
  },

  'Stochastic Random Sampler': { 
    type:'logic', 
    inputs:['probability_distribution', 'n_samples'], 
    outputs:['random_variants'], 
    logic:'Mersenne_Twister_PRNG', 
    function:'Lógica Probabilística', 
    synergies:['Monte Carlo Simulation', 'Transformer'], 
    desc:'Gera cenários aleatórios baseados em uma probabilidade. SINERGIA: Essencial para a Monte Carlo Simulation criar os 10.000 cenários de risco e para o Transformer lidar com incerteza.' 
  },
'100-Point Fast Track': { 
    type:'logic', 
    inputs:['total_confidence_score'], 
    outputs:['instant_trigger', 'bypass_signal'], 
    logic:'Score >= 100', 
    function:'Corte de Caminho', 
    synergies:['Safety Execution Gate', 'API Pusher'], 
    desc:'O Atalho do Sniper: Se o Cardus atingir 100 pontos, este bloco ignora o resto do processamento lento e envia a ordem imediatamente. Velocidade máxima, zero enrolação.' 
  },

  'Complexity Pruner': { 
    type:'logic', 
    inputs:['model_list', 'performance_metrics'], 
    outputs:['optimized_model_list'], 
    logic:'Feature_Importance_Threshold', 
    function:'Limpeza', 
    synergies:['Auto-Model Selector'], 
    desc:'Remove automaticamente do loop os modelos que estão atrasando o sistema e não estão contribuindo para os 100 pontos. Mantém o monstro leve e rápido.' 
  },

  'Confidence Saturation': { 
    type:'logic', 
    inputs:['probability_stream'], 
    outputs:['saturated_signal'], 
    logic:'Math.min(100, x)', 
    function:'Estabilização', 
    synergies:['Sigmoid Function'], 
    desc:'Evita que o sistema tente buscar "110% de certeza". Quando chega em 100, o sinal estabiliza. Isso impede que o ajuste fino mude uma decisão que já é perfeita.' 
  },

  'Latency vs Accuracy Trade-off': { 
    type:'logic', 
    inputs:['current_latency', 'confidence_score'], 
    outputs:['execution_mode'], 
    logic:'Priority_Switch', 
    function:'Ajuste Dinâmico', 
    synergies:['Event Loop'], 
    desc:'Se o mercado está rápido demais, ele prefere um modelo de 90 pontos instantâneo do que um de 100 pontos que demora 2 segundos. O lucro está no timing.' 
  },
'Platt Scaling (Calibration)': { 
    type:'logic', 
    inputs:['raw_confidence_score'], 
    outputs:['calibrated_probability'], 
    logic:'1 / (1 + exp(A*x + B))', 
    function:'Calibração de Certeza', 
    synergies:['Neural Network', 'Support Vector Machine'], 
    desc:'O Calibrador: Transforma o score bruto em uma probabilidade real. Ele garante que, se o Cardus diz "90%", ele realmente acerte 9 de cada 10 vezes. Ajusta a "arrogância" da IA.' 
  },

  'Adaptive Thresholding': { 
    type:'logic', 
    inputs:['confidence_score', 'market_volatility'], 
    outputs:['dynamic_trigger_level'], 
    logic:'Base_Threshold * (1 + Vol_Multiplier)', 
    function:'Ajuste de Sarrafo', 
    synergies:['Volatility Analysis', 'Safety Execution Gate'], 
    desc:'Ajusta o nível de exigência. Se o mercado está calmo, o gatilho pode ser 90 pontos. Se o mercado está um caos, ele sobe o sarrafo para 100 automaticamente para evitar falsos positivos.' 
  },

  'Hysteresis Filter (Signal Smoothing)': { 
    type:'logic', 
    inputs:['probability_stream'], 
    outputs:['stable_signal'], 
    logic:'Upper_Bound / Lower_Bound_Latch', 
    function:'Amortecedor de Sinal', 
    synergies:['Bayesian Updater', 'Live API Connector'], 
    desc:'Evita o "pisca-pisca". Se o sinal bateu 100 pontos, este bloco segura a decisão mesmo que ele caia para 98 por um milissegundo, evitando que o sniper cancele o tiro por causa de um ruído bobo.' 
  },

  'Weight Decay (Feature Pruning)': { 
    type:'logic', 
    inputs:['feature_importance_list'], 
    outputs:['optimized_weights'], 
    logic:'w = w * (1 - λ)', 
    function:'Poda de Irrelevância', 
    synergies:['Auto-Model Selector', 'Reinforcement Learning'], 
    desc:'Lógica Anti-Linguiça: Diminui gradualmente a importância de variáveis que não estão ajudando a chegar nos 100 pontos. Mantém o modelo focado apenas no "filé" dos dados.' 
  },

  'Bias Compensator': { 
    type:'logic', 
    inputs:['prediction', 'historical_error_bias'], 
    outputs:['unbiased_prediction'], 
    logic:'x - average_error', 
    function:'Correção de Vício', 
    synergies:['Quick Backtester', 'Reinforcement Learning'], 
    desc:'Corrige erros sistemáticos. Se o monstro sempre erra por 2% para cima em certos horários, este bloco subtrai esses 2% antes do resultado final. É a "mira compensada" para o vento.' 
  },

  'Micro-Momentum Validator': { 
    type:'logic', 
    inputs:['prediction_trend'], 
    outputs:['is_solid_100'], 
    logic:'dy/dt > 0', 
    function:'Validador de Inércia', 
    synergies:['Polynomial Regression', 'MACD'], 
    desc:'Verifica se os 100 pontos são sustentáveis. Ele só libera o tiro se os 100 pontos chegaram com uma tendência de subida sólida, e não por causa de um pico aleatório e passageiro (Spike).' 
  },
 'Prediction vs Reality (Ground Truth)': { 
    type:'logic', 
    inputs:['model_prediction', 'actual_outcome'], 
    outputs:['error_magnitude', 'accuracy_delta'], 
    logic:'MAPE / Log-Loss Analysis', 
    function:'Cálculo de Desvio', 
    synergies:['Reinforcement Learning', 'Quick Backtester'], 
    desc:'O Confronto Final: Compara o que o Cardus previu com o que realmente aconteceu. Ele gera o "erro real" que será usado para recalibrar os 55 modelos.' 
  },

  'Latency Autopsy': { 
    type:'logic', 
    inputs:['decision_timestamp', 'execution_timestamp', 'market_change_timestamp'], 
    outputs:['execution_lag', 'opportunity_loss'], 
    logic:'T2 - T1 = Δt', 
    function:'Análise de Velocidade', 
    synergies:['Event Loop', 'WebSocket Stream'], 
    desc:'A Autópsia do Tempo: Verifica se você perdeu dinheiro porque o sistema demorou para processar. Se a latência foi alta, ele sugere desativar blocos de "linguiça".' 
  },

  'Slippage Tracker (Odd/Price Decay)': { 
    type:'logic', 
    inputs:['predicted_odds', 'executed_odds'], 
    outputs:['slippage_value'], 
    logic:'(Executed - Predicted) / Predicted', 
    function:'Análise de Derrapagem', 
    synergies:['API Pusher (Webhook)', 'Expected Value (EV+)'], 
    desc:'Mede a "derrapagem": Se você previu uma odd 2.0, mas o mercado mudou para 1.8 antes do tiro chegar. Ajuda a ajustar o "fator de segurança" do Expected Value.' 
  },

  'Counterfactual Simulation (What-If)': { 
    type:'model', 
    inputs:['event_data', 'alternative_logic'], 
    outputs:['better_path_results'], 
    logic:'Stochastic_Replay', 
    function:'Simulação de Alternativas', 
    synergies:['Monte Carlo Simulation', 'Auto-Model Selector'], 
    desc:'O "E se...?": Simula se o resultado teria sido melhor se você tivesse usado outro dos seus 55 modelos. Se o Transformer teria ganho onde a Poisson perdeu, ele avisa.' 
  },

  'Model Attribution Score': { 
    type:'logic', 
    inputs:['winning_signal', 'model_votes'], 
    outputs:['model_reputation_update'], 
    logic:'Brier_Score / Shapley_Value', 
    function:'Atribuição de Crédito', 
    synergies:['Consensus Engine (The Judge)', 'Reinforcement Learning'], 
    desc:'Quem acertou?: Identifica qual dos 223 blocos foi o responsável pelo sucesso ou pelo erro. Aumenta ou diminui a "reputação" do modelo dentro do seletor automático.' 
  },

  'Automated Incident Report': { 
    type:'output', 
    inputs:['post_mortem_data', 'pnl_result'], 
    outputs:['summary_json', 'telegram_alert'], 
    logic:'Text_Template_Generator', 
    function:'Relatório Final', 
    synergies:['JSON Export', 'Telegram Bot'], 
    desc:'O Relato de Combate: Resume tudo: por que o sniper atirou, quanto tempo demorou, quem foi o modelo mestre e qual o erro final. Envia direto para o seu dashboard.' 
  },
 'Sentiment Vector Store': { 
    type:'data', 
    inputs:['text_data', 'sentiment_score'], 
    outputs:['historical_sentiment_context'], 
    logic:'Vector_Database (FAISS/Pinecone)', 
    function:'Memória Semântica', 
    synergies:['LLM Sentiment Analysis', 'Post-Mortem Analysis', 'Bayesian Updater'], 
    desc:'O Arquivo de Emoções: Guarda não apenas o score, mas o contexto das notícias. SINERGIA: O Bayesian Updater usa isso para saber se uma odd está caindo por causa de uma lesão real ou apenas por especulação de rede social.' 
  },

  'News Impact Decay': { 
    type:'logic', 
    inputs:['news_timestamp', 'initial_impact'], 
    outputs:['current_impact_weight'], 
    logic:'Exponential_Time_Decay', 
    function:'Validação de Frescor', 
    synergies:['Sentiment Vector Store', 'Time-Decay Weighting'], 
    desc:'Filtro de Relevância: Uma notícia de 2 horas atrás vale menos que uma de 2 minutos. Ele reduz o peso do sentimento conforme o tempo passa, evitando que o monstro reaja a "notícias velhas".' 
  },

  'Mood-Price Correlation': { 
    type:'model', 
    inputs:['sentiment_stream', 'price_odds_stream'], 
    outputs:['sentiment_lead_indicator'], 
    logic:'Granger_Causality / Pearson', 
    function:'Análise de Causa', 
    synergies:['Standard Deviation (Z-Score)', 'Expected Value (EV+)'], 
    desc:'Detector de Hype: Analisa se o sentimento positivo realmente precede uma subida de preço/odd. Se o sentimento sobe mas o preço não, ele avisa que pode ser uma armadilha (Fake News).' 
  },

  // ========== PROTEÇÃO DE BANCA (capital_defense) ==========

  'Cumulative Loss Guard (Hard Stop)': { 
    type:'logic', 
    inputs:['daily_pnl', 'max_loss_limit'], 
    outputs:['kill_signal'], 
    logic:'Threshold_Gate', 
    function:'Proteção Financeira', 
    synergies:['Real-time PnL Tracker', 'Emergency Kill Switch'], 
    desc:'O Teto de Perda: Se o monstro perder X% da banca no dia, este bloco corta a conexão do API Pusher instantaneamente. Sem exceções. É o freio de segurança final.' 
  },

  'Profit Lock (Trailing Take-Profit)': { 
    type:'logic', 
    inputs:['current_profit', 'target_profit'], 
    outputs:['withdraw_signal', 'lock_signal'], 
    logic:'Trailing_Stop_Logic', 
    function:'Garantia de Lucro', 
    synergies:['Execution Quota', 'Bankroll Protector'], 
    desc:'Cofre Automático: Se você atingiu 100% da meta e o lucro começar a cair, ele trava a operação e "garante o pão". Ele impede que você devolva o lucro do dia para o mercado.' 
  },

  'Exposure Diversifier': { 
    type:'logic', 
    inputs:['active_positions_list', 'new_snipe_request'], 
    outputs:['allow_exposure'], 
    logic:'Risk_Concentration_Check', 
    function:'Gestão de Exposição', 
    synergies:['Risk Analysis', 'Kelly Criterion'], 
    desc:'Anti-All-In: Impede que o monstro coloque muito dinheiro em um único evento ou em eventos correlacionados. Ele obriga o sistema a espalhar o risco para proteger a banca principal.' 
  },
'Real-time PnL Tracker': { 
    type:'output', 
    inputs:['transaction_id', 'actual_result', 'bet_amount'], 
    outputs:['daily_pnl', 'equity_curve', 'total_balance'], 
    logic:'Financial_Balance_Sheet', 
    function:'Rastreador Financeiro', 
    synergies:['Cumulative Loss Guard (Hard Stop)', 'Reward Calculator'], 
    desc:'O Contador: Registra cada ganho e perda em tempo real. Ele envia o saldo atualizado (equity_curve) para o Protetor de Banca decidir se continua operando.' 
  },

  'Bankroll Protector': { 
    type:'logic', 
    inputs:['equity_curve', 'total_balance'], 
    outputs:['safe_bet_limit', 'bankroll_health_status'], 
    logic:'Capital_Allocation_Strategy', 
    function:'Gestor de Cofre', 
    synergies:['Kelly Criterion', 'Execution Quota'], 
    desc:'O Guardião: Analisa a saúde da sua banca. Se o saldo cair muito, ele diminui automaticamente o valor das ordens no Execution Quota e no Kelly.' 
  },

  'Emergency Kill Switch': { 
    type:'logic', 
    inputs:['kill_signal', 'user_manual_stop', 'system_error'], 
    outputs:['MASTER_STOP'], 
    logic:'Safety_Relay', 
    function:'Corte Total', 
    synergies:['API Pusher (Webhook)', 'Event Loop'], 
    desc:'O Botão de Pânico: Um interruptor mestre. Se receber sinal do PnL Tracker (perda máxima) ou erro de sistema, ele mata o Event Loop e bloqueia o API Pusher na hora.' 
  },

  'Expected Value (EV+)': { 
    type:'model', 
    inputs:['true_probability', 'market_odds'], 
    outputs:['ev_score', 'is_profitable'], 
    logic:'(Prob * Odds) - 1', 
    function:'Validador de Lucro', 
    synergies:['Kelly Criterion', 'Safety Execution Gate'], 
    desc:'O Filtro de Valor: Calcula se a aposta paga mais do que o risco. Só envia o sinal de "is_profitable" se o retorno matemático for positivo.' 
  },

  'Standard Deviation (Z-Score)': { 
    type:'preprocessing', 
    inputs:['market_data_stream'], 
    outputs:['z_score_value', 'is_anomaly'], 
    logic:'(x - mean) / std_dev', 
    function:'Detector de Anomalia', 
    synergies:['Mood-Price Correlation', 'Outlier Sanitizer'], 
    desc:'O Radar de Estranheza: Mede o quanto o preço atual foge do normal. Ajuda o Mood-Price a saber se uma notícia causou um movimento desproporcional no mercado.' 
  },
 'Market Weather Station': { 
    type:'logic', 
    inputs:['market_volatility', 'sentiment_index', 'success_rate'], 
    outputs:['strategy_profile'], // Retorna: 'Aggressive', 'Standard', 'Conservative', 'Ghost'
    logic:'Macro_Environmental_Check', 
    function:'Climatologia de Mercado', 
    synergies:['Adaptive Thresholding', 'Risk Analysis', 'Auto-Model Selector'], 
    desc:'O Meteorologista: Analisa o "clima" geral do mercado. Se o dia está péssimo (muitas zebras), ele muda o perfil de todos os outros blocos para "Conservador" ou "Ghost" (apenas simulação), protegendo o sistema de tempestades financeiras.' 
  },

  'Sub-Millisecond Jitter Stabilizer': { 
    type:'logic', 
    inputs:['websocket_latency_stream'], 
    outputs:['stabilized_pulse'], 
    logic:'Time_Dilation_Buffer', 
    function:'Estabilização de Pulso', 
    synergies:['WebSocket Stream', 'Event Loop', 'Latency Monitor'], 
    desc:'O Estabilizador de Frequência: Lida com micro-atrasos (jitter) na internet. Ele garante que o Event Loop receba os dados em um ritmo constante, evitando que o sniper tome decisões baseadas em pacotes de dados que chegaram "encavalados" ou fora de ordem.' 
  },

  'AI Strategic Analyst (The Voice)': { 
    type:'output', 
    inputs:['post_mortem_data', 'strategy_profile', 'current_pnl'], 
    outputs:['human_readable_report', 'voice_alert'], 
    logic:'LLM_Natural_Language_Gen', 
    function:'Interface Executiva', 
    synergies:['Post-Mortem Analysis', 'Sentiment Database', 'Telegram Bot'], 
    desc:'A Voz do Monstro: Usa uma IA generativa para traduzir os 270 blocos de matemática em linguagem humana. Ele te envia áudios ou textos explicando: "Snipei agora porque o Time A cansou e a Odd estava descompensada em 15%". É o seu braço direito te dando o resumo da guerra.' 
  },
 'Stealth Fingerprint Rotator': { 
    type:'logic', 
    inputs:['api_request_stream'], 
    outputs:['camouflaged_request'], 
    logic:'Fingerprint_Randomizer / Proxy_Rotation', 
    function:'Camuflagem de Bot', 
    synergies:['API Pusher (Webhook)', 'Live API Connector'], 
    desc:'O Camaleão: Muda o "DNA" das suas requisições (User-Agent, IP, Latência). Ele faz o sniper parecer um usuário humano comum navegando devagar, evitando que as casas de apostas ou corretoras detectem e banam o seu bot.' 
  },

  'Genetic Hyper-Optimizer': { 
    type:'model', 
    inputs:['model_performance_history'], 
    outputs:['evolved_parameters'], 
    logic:'Genetic_Algorithm (Mutation/Crossover)', 
    function:'Evolução Darwiniana', 
    synergies:['Auto-Model Selector', 'Reinforcement Learning'], 
    desc:'O Evolucionista: Ele pega os melhores modelos, faz um "cruzamento" entre eles e cria uma nova geração de parâmetros. Ele "muta" a lógica da Poisson e da Regressão para encontrar fórmulas que nenhum humano conseguiria pensar.' 
  },

  'Cross-Market Arbitrage Link': { 
    type:'model', 
    inputs:['multi_source_odds', 'live_api_payload'], 
    outputs:['arb_opportunity_signal'], 
    logic:'Spread_Differential_Math', 
    function:'Detecção de Arbitragem', 
    synergies:['Expected Value (EV+)', 'Live API Connector'], 
    desc:'O Olho de Águia: Compara os preços entre diferentes APIs e fontes simultaneamente. Se a Casa A errou a odd e a Casa B está correta, ele identifica o lucro garantido (Surebet) entre elas antes que o mercado se ajuste.' 
  },

  'Quantum Risk Parity': { 
    type:'logic', 
    inputs:['market_volatility', 'bankroll_status'], 
    outputs:['optimized_position_size'], 
    logic:'Matrix_Covariance_Optimization', 
    function:'Equilíbrio de Exposição', 
    synergies:['Bankroll Protector', 'Risk Analysis'], 
    desc:'O Balancista: Em vez de apostar fixo, ele distribui o dinheiro baseado na "energia" do mercado. Se o risco subir 1%, ele reduz a posição em 1% instantaneamente. Ele mantém a banca em um equilíbrio perfeito, independente do caos lá fora.' 
  },
           'Linear Regression': { type: 'model', inputs: ['normalized_data'], outputs: ['regression_predictions'], logic: 'Linear', function: 'Regressão', dataType: 'continuous', synergies: ['Normalization'], incompatibilities: ['Random Forest'], desc: 'Regressão Linear' },
            'Ridge Regression': { type: 'model', inputs: ['normalized_data'], outputs: ['ridge_predictions'], logic: 'Ridge', function: 'Regressão', dataType: 'continuous', synergies: ['Normalization'], incompatibilities: [], desc: 'Ridge' },
            'Lasso Regression': { type: 'model', inputs: ['normalized_data'], outputs: ['lasso_predictions'], logic: 'Lasso', function: 'Regressão', dataType: 'continuous', synergies: ['Normalization'], incompatibilities: [], desc: 'Lasso' },
            'Elastic Net': { type: 'model', inputs: ['normalized_data'], outputs: ['elasticnet_predictions'], logic: 'ElasticNet', function: 'Regressão', dataType: 'continuous', synergies: ['Normalization'], incompatibilities: [], desc: 'Elastic Net' },
            'Polynomial Regression': { type: 'model', inputs: ['polynomial_features'], outputs: ['poly_predictions'], logic: 'PolyReg', function: 'Regressão', dataType: 'continuous', synergies: ['Polynomial Features'], incompatibilities: [], desc: 'Polinomial' },
            'Support Vector Regression': { type: 'model', inputs: ['normalized_data'], outputs: ['svr_predictions'], logic: 'SVR', function: 'Regressão', dataType: 'continuous', synergies: ['Normalization'], incompatibilities: [], desc: 'SVR' },
            'Logistic Regression': { type: 'model', inputs: ['normalized_data'], outputs: ['classification_predictions'], logic: 'Logistic', function: 'Classificação', dataType: 'classification', synergies: ['Normalization'], incompatibilities: [], desc: 'Logística' },
            'Softmax Regression': { type: 'model', inputs: ['normalized_data'], outputs: ['multiclass_predictions'], logic: 'Softmax', function: 'Classificação', dataType: 'multiclass', synergies: ['Normalization'], incompatibilities: [], desc: 'Softmax' },
            'Decision Tree': { type: 'model', inputs: ['clean_data'], outputs: ['tree_predictions'], logic: 'Tree', function: 'Classificação', dataType: 'classification', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Árvore' },
            'Random Forest': { type: 'model', inputs: ['clean_data'], outputs: ['ensemble_predictions'], logic: 'RF', function: 'Classificação', dataType: 'classification', synergies: ['Feature Selection'], incompatibilities: ['Linear Regression'], desc: 'Random Forest' },
            'Gradient Boosting': { type: 'model', inputs: ['clean_data'], outputs: ['boosting_predictions'], logic: 'GB', function: 'Classificação', dataType: 'classification', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Gradient Boosting' },
            'XGBoost': { type: 'model', inputs: ['clean_data'], outputs: ['xgb_predictions'], logic: 'XGB', function: 'Classificação', dataType: 'classification', synergies: ['Feature Selection'], incompatibilities: [], desc: 'XGBoost' },
            'LightGBM': { type: 'model', inputs: ['clean_data'], outputs: ['lgb_predictions'], logic: 'LGB', function: 'Classificação', dataType: 'classification', synergies: ['Feature Selection'], incompatibilities: [], desc: 'LightGBM' },
            'CatBoost': { type: 'model', inputs: ['clean_data'], outputs: ['cat_predictions'], logic: 'CatBoost', function: 'Classificação', dataType: 'classification', synergies: ['Feature Selection'], incompatibilities: [], desc: 'CatBoost' },
            'K-Nearest Neighbors': { type: 'model', inputs: ['normalized_data'], outputs: ['knn_predictions'], logic: 'KNN', function: 'Classificação', dataType: 'classification', synergies: ['Normalization'], incompatibilities: [], desc: 'KNN' },
            'Support Vector Machine': { type: 'model', inputs: ['normalized_data'], outputs: ['svm_predictions'], logic: 'SVM', function: 'Classificação', dataType: 'classification', synergies: ['Normalization'], incompatibilities: [], desc: 'SVM' },
            'Naive Bayes': { type: 'model', inputs: ['clean_data'], outputs: ['nb_predictions'], logic: 'NB', function: 'Classificação', dataType: 'classification', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Naive Bayes' },
            'Gaussian Naive Bayes': { type: 'model', inputs: ['normalized_data'], outputs: ['gnb_predictions'], logic: 'GNB', function: 'Classificação', dataType: 'classification', synergies: ['Normalization'], incompatibilities: [], desc: 'Gaussian NB' },
            'Multinomial Naive Bayes': { type: 'model', inputs: ['count_vectors'], outputs: ['mnb_predictions'], logic: 'MNB', function: 'Classificação', dataType: 'classification', synergies: ['Count Vectorizer'], incompatibilities: [], desc: 'Multinomial NB' },
            'Bernoulli Naive Bayes': { type: 'model', inputs: ['binary_data'], outputs: ['bnb_predictions'], logic: 'BNB', function: 'Classificação', dataType: 'classification', synergies: ['Binary Encoding'], incompatibilities: [], desc: 'Bernoulli NB' },
            'Neural Network': { type: 'model', inputs: ['normalized_data'], outputs: ['nn_predictions'], logic: 'NN', function: 'Modelagem', dataType: 'complex', synergies: ['Normalization'], incompatibilities: [], desc: 'Rede Neural' },
            'CNN': { type: 'model', inputs: ['image_data'], outputs: ['cnn_predictions'], logic: 'CNN', function: 'Visão', dataType: 'image', synergies: ['Image Processing'], incompatibilities: [], desc: 'CNN' },
            'RNN': { type: 'model', inputs: ['timeseries_data'], outputs: ['rnn_predictions'], logic: 'RNN', function: 'Sequência', dataType: 'sequence', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'RNN' },
            'LSTM': { type: 'model', inputs: ['timeseries_data'], outputs: ['lstm_predictions'], logic: 'LSTM', function: 'Sequência', dataType: 'sequence', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'LSTM' },
            'GRU': { type: 'model', inputs: ['timeseries_data'], outputs: ['gru_predictions'], logic: 'GRU', function: 'Sequência', dataType: 'sequence', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'GRU' },
            'Transformer': { type: 'model', inputs: ['sequence_data'], outputs: ['transformer_predictions'], logic: 'Transformer', function: 'Sequência', dataType: 'sequence', synergies: ['BERT Embeddings'], incompatibilities: [], desc: 'Transformer' },
            'Attention Mechanism': { type: 'model', inputs: ['sequence_data'], outputs: ['attention_output'], logic: 'Attention', function: 'Mecanismo', dataType: 'sequence', synergies: ['Transformer'], incompatibilities: [], desc: 'Attention' },
            'Seq2Seq': { type: 'model', inputs: ['sequence_data'], outputs: ['seq2seq_output'], logic: 'Seq2Seq', function: 'Sequência', dataType: 'sequence', synergies: ['LSTM'], incompatibilities: [], desc: 'Seq2Seq' },
            'Graph Neural Network': { type: 'model', inputs: ['graph_data'], outputs: ['gnn_predictions'], logic: 'GNN', function: 'Grafo', dataType: 'graph', synergies: ['Graph Data'], incompatibilities: [], desc: 'GNN' },
            'Graph Convolutional Network': { type: 'model', inputs: ['graph_data'], outputs: ['gcn_predictions'], logic: 'GCN', function: 'Grafo', dataType: 'graph', synergies: ['Graph Data'], incompatibilities: [], desc: 'GCN' },
            'K-Means': { type: 'model', inputs: ['normalized_data'], outputs: ['kmeans_clusters'], logic: 'KMeans', function: 'Clustering', dataType: 'clusters', synergies: ['Normalization'], incompatibilities: [], desc: 'K-Means' },
            'DBSCAN': { type: 'model', inputs: ['normalized_data'], outputs: ['dbscan_clusters'], logic: 'DBSCAN', function: 'Clustering', dataType: 'clusters', synergies: ['Normalization'], incompatibilities: [], desc: 'DBSCAN' },
            'Hierarchical Clustering': { type: 'model', inputs: ['normalized_data'], outputs: ['hierarchical_clusters'], logic: 'Hierarchical', function: 'Clustering', dataType: 'clusters', synergies: ['Normalization'], incompatibilities: [], desc: 'Hierárquico' },
            'Gaussian Mixture Model': { type: 'model', inputs: ['normalized_data'], outputs: ['gmm_clusters'], logic: 'GMM', function: 'Clustering', dataType: 'clusters', synergies: ['Normalization'], incompatibilities: [], desc: 'GMM' },
            'Spectral Clustering': { type: 'model', inputs: ['normalized_data'], outputs: ['spectral_clusters'], logic: 'Spectral', function: 'Clustering', dataType: 'clusters', synergies: ['Normalization'], incompatibilities: [], desc: 'Spectral' },
            'Isolation Forest': { type: 'model', inputs: ['tabular_data'], outputs: ['anomaly_scores'], logic: 'IsoForest', function: 'Anomalia', dataType: 'anomaly', synergies: ['Outlier Detection'], incompatibilities: [], desc: 'Isolation Forest' },
            'Local Outlier Factor': { type: 'model', inputs: ['normalized_data'], outputs: ['lof_scores'], logic: 'LOF', function: 'Anomalia', dataType: 'anomaly', synergies: ['Normalization'], incompatibilities: [], desc: 'LOF' },
            'One-Class SVM': { type: 'model', inputs: ['normalized_data'], outputs: ['ocsvm_scores'], logic: 'OCSVM', function: 'Anomalia', dataType: 'anomaly', synergies: ['Normalization'], incompatibilities: [], desc: 'One-Class SVM' },
            'Autoencoder Anomaly': { type: 'model', inputs: ['encoded_features'], outputs: ['ae_anomaly_scores'], logic: 'AEAnomaly', function: 'Anomalia', dataType: 'anomaly', synergies: ['Autoencoder'], incompatibilities: [], desc: 'AE Anomaly' },
            'Apriori': { type: 'model', inputs: ['transaction_data'], outputs: ['association_rules'], logic: 'Apriori', function: 'Associação', dataType: 'rules', synergies: ['Transaction Data'], incompatibilities: [], desc: 'Apriori' },
            'Eclat': { type: 'model', inputs: ['transaction_data'], outputs: ['itemsets'], logic: 'Eclat', function: 'Associação', dataType: 'itemsets', synergies: ['Transaction Data'], incompatibilities: [], desc: 'Eclat' },
            'Collaborative Filtering': { type: 'model', inputs: ['recommendation_data'], outputs: ['recommendations'], logic: 'ColFilter', function: 'Recomendação', dataType: 'recommendations', synergies: ['Recommendation Data'], incompatibilities: [], desc: 'Collab Filtering' },
            'Content-Based Filtering': { type: 'model', inputs: ['recommendation_data'], outputs: ['recommendations'], logic: 'ContentFilter', function: 'Recomendação', dataType: 'recommendations', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Content Filter' },
            'Matrix Factorization': { type: 'model', inputs: ['recommendation_data'], outputs: ['latent_factors'], logic: 'MatFactor', function: 'Recomendação', dataType: 'factors', synergies: ['Recommendation Data'], incompatibilities: [], desc: 'Matrix Factor' },
            'Deep Learning': { type: 'model', inputs: ['normalized_data'], outputs: ['deep_predictions'], logic: 'DL', function: 'Modelagem', dataType: 'complex', synergies: ['Neural Network'], incompatibilities: [], desc: 'Deep Learning' },
            'Reinforcement Learning': { type: 'model', inputs: ['state_data'], outputs: ['rl_actions'], logic: 'RL', function: 'Controle', dataType: 'actions', synergies: ['State Data'], incompatibilities: [], desc: 'RL' },
            'Q-Learning': { type: 'model', inputs: ['state_data'], outputs: ['q_values'], logic: 'QLearning', function: 'RL', dataType: 'values', synergies: ['State Data'], incompatibilities: [], desc: 'Q-Learning' },
            'Policy Gradient': { type: 'model', inputs: ['state_data'], outputs: ['policy_output'], logic: 'PolicyGrad', function: 'RL', dataType: 'policy', synergies: ['State Data'], incompatibilities: [], desc: 'Policy Gradient' },
            'Actor-Critic': { type: 'model', inputs: ['state_data'], outputs: ['ac_output'], logic: 'ActorCritic', function: 'RL', dataType: 'policy', synergies: ['State Data'], incompatibilities: [], desc: 'Actor-Critic' },
            'Genetic Algorithm': { type: 'model', inputs: ['population_data'], outputs: ['evolved_solution'], logic: 'GA', function: 'Otimização', dataType: 'solution', synergies: ['Population Data'], incompatibilities: [], desc: 'Genético' },
            'Particle Swarm': { type: 'model', inputs: ['optimization_data'], outputs: ['optimized_params'], logic: 'PSO', function: 'Otimização', dataType: 'params', synergies: ['Optimization Data'], incompatibilities: [], desc: 'PSO' },
            'Simulated Annealing': { type: 'model', inputs: ['optimization_data'], outputs: ['optimized_solution'], logic: 'SA', function: 'Otimização', dataType: 'solution', synergies: ['Optimization Data'], incompatibilities: [], desc: 'Simulated Annealing' },
            'Bayesian Optimization': { type: 'model', inputs: ['optimization_data'], outputs: ['optimal_params'], logic: 'BayesOpt', function: 'Otimização', dataType: 'params', synergies: ['Optimization Data'], incompatibilities: [], desc: 'Bayes Opt' },
            'Ensemble Methods': { type: 'model', inputs: ['predictions_list'], outputs: ['ensemble_output'], logic: 'Ensemble', function: 'Combinação', dataType: 'ensemble', synergies: ['Voting'], incompatibilities: [], desc: 'Ensemble' },
            'Voting Classifier': { type: 'model', inputs: ['predictions_list'], outputs: ['voted_predictions'], logic: 'Voting', function: 'Combinação', dataType: 'classification', synergies: ['Ensemble Methods'], incompatibilities: [], desc: 'Voting' },
            'Stacking': { type: 'model', inputs: ['predictions_list'], outputs: ['stacked_predictions'], logic: 'Stacking', function: 'Combinação', dataType: 'predictions', synergies: ['Ensemble Methods'], incompatibilities: [], desc: 'Stacking' },
            'Bagging': { type: 'model', inputs: ['clean_data'], outputs: ['bagging_predictions'], logic: 'Bagging', function: 'Ensemble', dataType: 'classification', synergies: ['Random Forest'], incompatibilities: [], desc: 'Bagging' },
            'AdaBoost': { type: 'model', inputs: ['clean_data'], outputs: ['adaboost_predictions'], logic: 'AdaBoost', function: 'Boosting', dataType: 'classification', synergies: ['Decision Tree'], incompatibilities: [], desc: 'AdaBoost' },
            'Time Series ARIMA': { type: 'model', inputs: ['timeseries_data'], outputs: ['arima_forecast'], logic: 'ARIMA', function: 'Previsão', dataType: 'forecast', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'ARIMA' },
            'SARIMA': { type: 'model', inputs: ['timeseries_data'], outputs: ['sarima_forecast'], logic: 'SARIMA', function: 'Previsão', dataType: 'forecast', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'SARIMA' },
            'Prophet': { type: 'model', inputs: ['timeseries_data'], outputs: ['prophet_forecast'], logic: 'Prophet', function: 'Previsão', dataType: 'forecast', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Prophet' },
            'Exponential Smoothing': { type: 'model', inputs: ['timeseries_data'], outputs: ['smoothed_forecast'], logic: 'ExpSmooth', function: 'Previsão', dataType: 'forecast', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Exp Smoothing' },
            'Kalman Filter': { type: 'model', inputs: ['timeseries_data'], outputs: ['filtered_signal'], logic: 'Kalman', function: 'Filtragem', dataType: 'filtered', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Kalman' },
            'Hidden Markov Model': { type: 'model', inputs: ['sequence_data'], outputs: ['hmm_output'], logic: 'HMM', function: 'Sequência', dataType: 'sequence', synergies: ['Sequence Data'], incompatibilities: [], desc: 'HMM' },
            'Conditional Random Field': { type: 'model', inputs: ['sequence_data'], outputs: ['crf_output'], logic: 'CRF', function: 'Sequência', dataType: 'sequence', synergies: ['NLP'], incompatibilities: [], desc: 'CRF' },
            'Variational Autoencoder': { type: 'model', inputs: ['normalized_data'], outputs: ['vae_output'], logic: 'VAE', function: 'Geração', dataType: 'generated', synergies: ['Normalization'], incompatibilities: [], desc: 'VAE' },
            'Generative Adversarial Network': { type: 'model', inputs: ['noise_data'], outputs: ['generated_data'], logic: 'GAN', function: 'Geração', dataType: 'generated', synergies: ['Image Data'], incompatibilities: [], desc: 'GAN' },
            'Diffusion Model': { type: 'model', inputs: ['image_data'], outputs: ['diffusion_output'], logic: 'Diffusion', function: 'Geração', dataType: 'generated', synergies: ['Image Data'], incompatibilities: [], desc: 'Diffusion' },
            'Stable Diffusion': { type: 'model', inputs: ['text_data'], outputs: ['generated_images'], logic: 'StableDiff', function: 'Geração', dataType: 'image', synergies: ['Text Data'], incompatibilities: [], desc: 'Stable Diffusion' },
            'Large Language Model': { type: 'model', inputs: ['text_data'], outputs: ['llm_output'], logic: 'LLM', function: 'Geração', dataType: 'text', synergies: ['Text Data'], incompatibilities: [], desc: 'LLM' },
            'GPT': { type: 'model', inputs: ['text_data'], outputs: ['gpt_output'], logic: 'GPT', function: 'Geração', dataType: 'text', synergies: ['Text Data'], incompatibilities: [], desc: 'GPT' },
            'BERT': { type: 'model', inputs: ['text_data'], outputs: ['bert_output'], logic: 'BERT', function: 'NLP', dataType: 'embeddings', synergies: ['Text Data'], incompatibilities: [], desc: 'BERT' },
            'T5': { type: 'model', inputs: ['text_data'], outputs: ['t5_output'], logic: 'T5', function: 'NLP', dataType: 'text', synergies: ['Text Data'], incompatibilities: [], desc: 'T5' },
            'ELECTRA': { type: 'model', inputs: ['text_data'], outputs: ['electra_output'], logic: 'ELECTRA', function: 'NLP', dataType: 'embeddings', synergies: ['Text Data'], incompatibilities: [], desc: 'ELECTRA' },
            'RoBERTa': { type: 'model', inputs: ['text_data'], outputs: ['roberta_output'], logic: 'RoBERTa', function: 'NLP', dataType: 'embeddings', synergies: ['Text Data'], incompatibilities: [], desc: 'RoBERTa' },
            'ALBERT': { type: 'model', inputs: ['text_data'], outputs: ['albert_output'], logic: 'ALBERT', function: 'NLP', dataType: 'embeddings', synergies: ['Text Data'], incompatibilities: [], desc: 'ALBERT' },
            'XLNet': { type: 'model', inputs: ['text_data'], outputs: ['xlnet_output'], logic: 'XLNet', function: 'NLP', dataType: 'embeddings', synergies: ['Text Data'], incompatibilities: [], desc: 'XLNet' },
            'DistilBERT': { type: 'model', inputs: ['text_data'], outputs: ['distilbert_output'], logic: 'DistilBERT', function: 'NLP', dataType: 'embeddings', synergies: ['Text Data'], incompatibilities: [], desc: 'DistilBERT' },
            'Sentiment Analysis': { type: 'model', inputs: ['text_data'], outputs: ['sentiment_scores'], logic: 'Sentiment', function: 'NLP', dataType: 'scores', synergies: ['Text Data'], incompatibilities: [], desc: 'Sentimento' },
            'Named Entity Recognition': { type: 'model', inputs: ['text_data'], outputs: ['entities'], logic: 'NER', function: 'NLP', dataType: 'entities', synergies: ['Text Data'], incompatibilities: [], desc: 'NER' },
            'Text Classification': { type: 'model', inputs: ['text_data'], outputs: ['text_classes'], logic: 'TextClass', function: 'NLP', dataType: 'classification', synergies: ['Text Data'], incompatibilities: [], desc: 'Text Class' },
            'Question Answering': { type: 'model', inputs: ['text_data'], outputs: ['answers'], logic: 'QA', function: 'NLP', dataType: 'text', synergies: ['Text Data'], incompatibilities: [], desc: 'QA' },
            'Machine Translation': { type: 'model', inputs: ['text_data'], outputs: ['translated_text'], logic: 'Translation', function: 'NLP', dataType: 'text', synergies: ['Text Data'], incompatibilities: [], desc: 'Tradução' },
            'Speech Recognition': { type: 'model', inputs: ['audio_data'], outputs: ['recognized_text'], logic: 'SpeechRec', function: 'Audio', dataType: 'text', synergies: ['Audio Data'], incompatibilities: [], desc: 'Speech Rec' },
            'Text-to-Speech': { type: 'model', inputs: ['text_data'], outputs: ['audio_output'], logic: 'TTS', function: 'Audio', dataType: 'audio', synergies: ['Text Data'], incompatibilities: [], desc: 'TTS' },
            'Object Detection': { type: 'model', inputs: ['image_data'], outputs: ['detected_objects'], logic: 'ObjDetect', function: 'Visão', dataType: 'objects', synergies: ['Image Data'], incompatibilities: [], desc: 'Object Detect' },
            'Image Segmentation': { type: 'model', inputs: ['image_data'], outputs: ['segmented_image'], logic: 'Segment', function: 'Visão', dataType: 'segmentation', synergies: ['Image Data'], incompatibilities: [], desc: 'Segmentação' },
            'Semantic Segmentation': { type: 'model', inputs: ['image_data'], outputs: ['semantic_segments'], logic: 'SemanticSeg', function: 'Visão', dataType: 'segmentation', synergies: ['Image Data'], incompatibilities: [], desc: 'Semantic Seg' },
            'Instance Segmentation': { type: 'model', inputs: ['image_data'], outputs: ['instance_segments'], logic: 'InstanceSeg', function: 'Visão', dataType: 'segmentation', synergies: ['Image Data'], incompatibilities: [], desc: 'Instance Seg' },
            'Panoptic Segmentation': { type: 'model', inputs: ['image_data'], outputs: ['panoptic_segments'], logic: 'PanopticSeg', function: 'Visão', dataType: 'segmentation', synergies: ['Image Data'], incompatibilities: [], desc: 'Panoptic Seg' },
            'Pose Estimation': { type: 'model', inputs: ['image_data'], outputs: ['pose_keypoints'], logic: 'Pose', function: 'Visão', dataType: 'keypoints', synergies: ['Image Data'], incompatibilities: [], desc: 'Pose' },
            'Action Recognition': { type: 'model', inputs: ['video_data'], outputs: ['recognized_actions'], logic: 'Action', function: 'Visão', dataType: 'actions', synergies: ['Video Data'], incompatibilities: [], desc: 'Action Rec' },
            'Video Classification': { type: 'model', inputs: ['video_data'], outputs: ['video_classes'], logic: 'VideoClass', function: 'Visão', dataType: 'classification', synergies: ['Video Data'], incompatibilities: [], desc: 'Video Class' },
            'Optical Flow': { type: 'model', inputs: ['video_data'], outputs: ['flow_vectors'], logic: 'OpticalFlow', function: 'Visão', dataType: 'vectors', synergies: ['Video Data'], incompatibilities: [], desc: 'Optical Flow' },
            'Depth Estimation': { type: 'model', inputs: ['image_data'], outputs: ['depth_map'], logic: 'Depth', function: 'Visão', dataType: 'depth', synergies: ['Image Data'], incompatibilities: [], desc: 'Depth' },
            '3D Reconstruction': { type: 'model', inputs: ['image_data'], outputs: ['3d_model'], logic: '3DRecon', function: 'Visão', dataType: '3d', synergies: ['Image Data'], incompatibilities: [], desc: '3D Recon' },
            'Face Recognition': { type: 'model', inputs: ['image_data'], outputs: ['face_embeddings'], logic: 'FaceRec', function: 'Visão', dataType: 'embeddings', synergies: ['Image Data'], incompatibilities: [], desc: 'Face Rec' },
            'Face Detection': { type: 'model', inputs: ['image_data'], outputs: ['face_locations'], logic: 'FaceDetect', function: 'Visão', dataType: 'locations', synergies: ['Image Data'], incompatibilities: [], desc: 'Face Detect' },
            'Facial Expression': { type: 'model', inputs: ['image_data'], outputs: ['expressions'], logic: 'Expression', function: 'Visão', dataType: 'classification', synergies: ['Image Data'], incompatibilities: [], desc: 'Expression' },
            'Emotion Recognition': { type: 'model', inputs: ['image_data'], outputs: ['emotions'], logic: 'Emotion', function: 'Análise', dataType: 'classification', synergies: ['Image Data'], incompatibilities: [], desc: 'Emotion' },
            'Gaze Estimation': { type: 'model', inputs: ['image_data'], outputs: ['gaze_direction'], logic: 'Gaze', function: 'Visão', dataType: 'direction', synergies: ['Image Data'], incompatibilities: [], desc: 'Gaze' },
            'Hand Gesture': { type: 'model', inputs: ['image_data'], outputs: ['gesture_class'], logic: 'Gesture', function: 'Visão', dataType: 'classification', synergies: ['Image Data'], incompatibilities: [], desc: 'Gesture' },
            'Crowd Counting': { type: 'model', inputs: ['image_data'], outputs: ['crowd_count'], logic: 'Crowd', function: 'Visão', dataType: 'count', synergies: ['Image Data'], incompatibilities: [], desc: 'Crowd Count' },
            'Scene Understanding': { type: 'model', inputs: ['image_data'], outputs: ['scene_desc'], logic: 'Scene', function: 'Visão', dataType: 'desc', synergies: ['Image Data'], incompatibilities: [], desc: 'Scene' },
            'Visual Question Answering': { type: 'model', inputs: ['image_data'], outputs: ['vqa_answer'], logic: 'VQA', function: 'Multimodal', dataType: 'text', synergies: ['Image Data'], incompatibilities: [], desc: 'VQA' },
            'Image Captioning': { type: 'model', inputs: ['image_data'], outputs: ['captions'], logic: 'Caption', function: 'Visão', dataType: 'text', synergies: ['Image Data'], incompatibilities: [], desc: 'Captioning' },
            'Cross-Modal Retrieval': { type: 'model', inputs: ['image_data'], outputs: ['retrieved_items'], logic: 'CrossModal', function: 'Multimodal', dataType: 'results', synergies: ['Image Data'], incompatibilities: [], desc: 'Cross-Modal' },
            'Anomaly Detection': { type: 'model', inputs: ['tabular_data'], outputs: ['anomaly_flags'], logic: 'Anomaly', function: 'Detecção', dataType: 'flags', synergies: ['Isolation Forest'], incompatibilities: [], desc: 'Anomalia' },
            'Fraud Detection': { type: 'model', inputs: ['transaction_data'], outputs: ['fraud_scores'], logic: 'Fraud', function: 'Detecção', dataType: 'scores', synergies: ['Anomaly Detection'], incompatibilities: [], desc: 'Fraude' },
            'Churn Prediction': { type: 'model', inputs: ['customer_data'], outputs: ['churn_probability'], logic: 'Churn', function: 'Previsão', dataType: 'probability', synergies: ['Classification Models'], incompatibilities: [], desc: 'Churn' },
            'Demand Forecasting': { type: 'model', inputs: ['timeseries_data'], outputs: ['demand_forecast'], logic: 'Demand', function: 'Previsão', dataType: 'forecast', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Demand' },
            'Price Optimization': { type: 'model', inputs: ['market_data'], outputs: ['optimal_prices'], logic: 'Price', function: 'Otimização', dataType: 'prices', synergies: ['Market Data'], incompatibilities: [], desc: 'Price Opt' },
            'Customer Segmentation': { type: 'model', inputs: ['customer_data'], outputs: ['customer_segments'], logic: 'Segment', function: 'Clustering', dataType: 'segments', synergies: ['K-Means'], incompatibilities: [], desc: 'Segmentação' },
            'Lead Scoring': { type: 'model', inputs: [], outputs: ['lead_scores'], logic: 'LeadScore', function: 'Classificação', dataType: 'scores', synergies: ['Classification Models'], incompatibilities: [], desc: 'Lead Score' },
            'Recommendation Engine': { type: 'model', inputs: ['user_behavior'], outputs: ['recommendations'], logic: 'Recommend', function: 'Recomendação', dataType: 'recommendations', synergies: ['Collaborative Filtering'], incompatibilities: [], desc: 'Recomendação' },
            'Personalization': { type: 'model', inputs: ['user_data'], outputs: ['personalized_content'], logic: 'Personal', function: 'Customização', dataType: 'content', synergies: ['User Behavior'], incompatibilities: [], desc: 'Personalização' },
            'A/B Testing': { type: 'model', inputs: ['experiment_data'], outputs: ['test_results'], logic: 'ABTest', function: 'Análise', dataType: 'results', synergies: ['Statistical Analysis'], incompatibilities: [], desc: 'A/B Test' },
            'Causal Inference': { type: 'model', inputs: ['experiment_data'], outputs: ['causal_effects'], logic: 'Causal', function: 'Análise', dataType: 'effects', synergies: ['Statistical Analysis'], incompatibilities: [], desc: 'Causal' },

            // SAÍDA (20+ blocos)
            'JSON Export': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'JSON', function: 'Exportação', dataType: 'json', synergies: [], incompatibilities: [], desc: 'Exporta JSON' },
            'CSV Export': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'CSV', function: 'Exportação', dataType: 'csv', synergies: [], incompatibilities: [], desc: 'Exporta CSV' },
            'Visualization': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Viz', function: 'Visualização', dataType: 'visual', synergies: [], incompatibilities: [], desc: 'Visualização' },
            'Database Save': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'DBSave', function: 'Armazenamento', dataType: 'database', synergies: [], incompatibilities: [], desc: 'Banco de Dados' },
            'API Endpoint': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'API', function: 'Serviço', dataType: 'api', synergies: [], incompatibilities: [], desc: 'API' },
            'Dashboard': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Dashboard', function: 'Visualização', dataType: 'visual', synergies: [], incompatibilities: [], desc: 'Dashboard' },
            'Report Generation': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Report', function: 'Relatório', dataType: 'document', synergies: [], incompatibilities: [], desc: 'Relatório' },
            'Email Alert': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Email', function: 'Notificação', dataType: 'notification', synergies: [], incompatibilities: [], desc: 'Email' },
            'Slack Notification': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Slack', function: 'Notificação', dataType: 'notification', synergies: [], incompatibilities: [], desc: 'Slack' },
            'Cloud Storage': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Cloud', function: 'Armazenamento', dataType: 'cloud', synergies: [], incompatibilities: [], desc: 'Cloud' },
            'Real-time Stream': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Stream', function: 'Streaming', dataType: 'stream', synergies: [], incompatibilities: [], desc: 'Streaming' },
            'Model Registry': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Registry', function: 'Armazenamento', dataType: 'model', synergies: [], incompatibilities: [], desc: 'Registry' },
            'Metrics Logger': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Metrics', function: 'Logging', dataType: 'metrics', synergies: [], incompatibilities: [], desc: 'Metrics' },
            'Webhook': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Webhook', function: 'Integração', dataType: 'http', synergies: [], incompatibilities: [], desc: 'Webhook' },
            'File Archive': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Archive', function: 'Armazenamento', dataType: 'archive', synergies: [], incompatibilities: [], desc: 'Arquivo' },
            'Parquet Output': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Parquet', function: 'Exportação', dataType: 'parquet', synergies: [], incompatibilities: [], desc: 'Parquet' },
            'HDF5 Output': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'HDF5', function: 'Exportação', dataType: 'hdf5', synergies: [], incompatibilities: [], desc: 'HDF5' },
            'Excel Export': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Excel', function: 'Exportação', dataType: 'excel', synergies: [], incompatibilities: [], desc: 'Excel' },
            'PDF Report': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'PDF', function: 'Relatório', dataType: 'pdf', synergies: [], incompatibilities: [], desc: 'PDF' },
            'Model Deployment': { type: 'output', inputs: ['predictions_list'], outputs: [], logic: 'Deploy', function: 'Deployment', dataType: 'model', Synergies: [], Incompatibilities: [], desc: 'Deploy' },
           'XML Input': { type: 'data', inputs: [], outputs: ['xml_data'], logic: 'XML', function: 'Leitura', dataType: 'xml', synergies: ['Data Parsing'], incompatibilities: [], desc: 'Importa XML' },
            'API REST': { type: 'data', inputs: [], outputs: ['api_data'], logic: 'REST', function: 'Coleta', dataType: 'json', synergies: ['Data Parsing'], incompatibilities: [], desc: 'API REST' },
            'GraphQL API': { type: 'data', inputs: [], outputs: ['graphql_data'], logic: 'GraphQL', function: 'Coleta', dataType: 'json', synergies: ['Data Parsing'], incompatibilities: [], desc: 'GraphQL' },
            'Database SQL': { type: 'data', inputs: [], outputs: ['sql_data'], logic: 'SQL', function: 'Leitura', dataType: 'structured', synergies: ['Cleaning'], incompatibilities: [], desc: 'Banco SQL' },
            'MongoDB': { type: 'data', inputs: [], outputs: ['nosql_data'], logic: 'NoSQL', function: 'Leitura', dataType: 'document', synergies: ['Data Parsing'], incompatibilities: [], desc: 'MongoDB' },
            'Redis Cache': { type: 'data', inputs: [], outputs: ['cached_data'], logic: 'Cache', function: 'Leitura', dataType: 'cached', synergies: ['Data Parsing'], incompatibilities: [], desc: 'Redis' },
            'Elasticsearch': { type: 'data', inputs: [], outputs: ['search_data'], logic: 'Search', function: 'Busca', dataType: 'indexed', synergies: ['Cleaning'], incompatibilities: [], desc: 'Elasticsearch' },
            'BigQuery': { type: 'data', inputs: [], outputs: ['bigdata'], logic: 'BigQuery', function: 'Leitura', dataType: 'bigdata', synergies: ['Aggregation'], incompatibilities: [], desc: 'BigQuery' },
            'Parquet File': { type: 'data', inputs: [], outputs: ['columnar_data'], logic: 'Parquet', function: 'Leitura', dataType: 'columnar', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Parquet' },
            'HDF5 File': { type: 'data', inputs: [], outputs: ['hierarchical_data'], logic: 'HDF5', function: 'Leitura', dataType: 'hierarchical', synergies: ['Feature Selection'], incompatibilities: [], desc: 'HDF5' },
            'Image Dataset': { type: 'data', inputs: [], outputs: ['image_data'], logic: 'Images', function: 'Leitura', dataType: 'image', synergies: ['Image Preprocessing', 'CNN'], incompatibilities: [], desc: 'Imagens' },
            'Video Dataset': { type: 'data', inputs: [], outputs: ['video_data'], logic: 'Videos', function: 'Leitura', dataType: 'video', synergies: ['Video Preprocessing'], incompatibilities: [], desc: 'Vídeos' },
            'Audio Dataset': { type: 'data', inputs: [], outputs: ['audio_data'], logic: 'Audio', function: 'Leitura', dataType: 'audio', synergies: ['Audio Preprocessing'], incompatibilities: [], desc: 'Áudio' },
            'Text Dataset': { type: 'data', inputs: [], outputs: ['text_data'], logic: 'Text', function: 'Leitura', dataType: 'text', synergies: ['Tokenization', 'NLP'], incompatibilities: [], desc: 'Textos' },
            'Time Series Data': { type: 'data', inputs: [], outputs: ['timeseries_data'], logic: 'TimeSeries', function: 'Leitura', dataType: 'timeseries', synergies: ['Detrending', 'RNN'], incompatibilities: [], desc: 'Séries Temporais' },
            'Streaming Data': { type: 'data', inputs: [], outputs: ['stream_data'], logic: 'Stream', function: 'Coleta', dataType: 'stream', synergies: ['Real-time'], incompatibilities: [], desc: 'Streaming' },
            'Kafka Stream': { type: 'data', inputs: [], outputs: ['kafka_data'], logic: 'Kafka', function: 'Coleta', dataType: 'stream', synergies: ['Real-time'], incompatibilities: [], desc: 'Kafka' },
            'AWS S3': { type: 'data', inputs: [], outputs: ['cloud_data'], logic: 'S3', function: 'Leitura', dataType: 'cloud', synergies: ['Cleaning'], incompatibilities: [], desc: 'AWS S3' },
            'Google Cloud': { type: 'data', inputs: [], outputs: ['cloud_data'], logic: 'GCS', function: 'Leitura', dataType: 'cloud', synergies: ['Cleaning'], incompatibilities: [], desc: 'GCS' },
            'Azure Blob': { type: 'data', inputs: [], outputs: ['cloud_data'], logic: 'Azure', function: 'Leitura', dataType: 'cloud', synergies: ['Cleaning'], incompatibilities: [], desc: 'Azure' },
            'Web Scraping': { type: 'data', inputs: [], outputs: ['scraped_data'], logic: 'Scrape', function: 'Coleta', dataType: 'web', synergies: ['Cleaning'], incompatibilities: [], desc: 'Web Scraping' },
            'PDF Extract': { type: 'data', inputs: [], outputs: ['pdf_data'], logic: 'PDF', function: 'Leitura', dataType: 'document', synergies: ['Text Processing'], incompatibilities: [], desc: 'PDF' },
            'Excel File': { type: 'data', inputs: [], outputs: ['spreadsheet_data'], logic: 'Excel', function: 'Leitura', dataType: 'tabular', synergies: ['Cleaning'], incompatibilities: [], desc: 'Excel' },
            'Google Sheets': { type: 'data', inputs: [], outputs: ['sheets_data'], logic: 'Sheets', function: 'Leitura', dataType: 'tabular', synergies: ['Cleaning'], incompatibilities: [], desc: 'Sheets' },
            'Synthetic Data': { type: 'data', inputs: [], outputs: ['synthetic_data'], logic: 'Synthetic', function: 'Geração', dataType: 'synthetic', synergies: ['Testing'], incompatibilities: [], desc: 'Sintético' },
            'Random Generator': { type: 'data', inputs: [], outputs: ['random_data'], logic: 'Random', function: 'Geração', dataType: 'random', synergies: ['Testing'], incompatibilities: [], desc: 'Aleatório' },
            'Sensor Data': { type: 'data', inputs: [], outputs: ['sensor_data'], logic: 'Sensors', function: 'Coleta', dataType: 'sensor', synergies: ['Denoising'], incompatibilities: [], desc: 'Sensores' },
            'GPS Data': { type: 'data', inputs: [], outputs: ['gps_data'], logic: 'GPS', function: 'Coleta', dataType: 'geospatial', synergies: ['Geospatial'], incompatibilities: [], desc: 'GPS' },
            'Social Media': { type: 'data', inputs: [], outputs: ['social_data'], logic: 'Social', function: 'Coleta', dataType: 'text', synergies: ['Text Processing'], incompatibilities: [], desc: 'Redes Sociais' },
            'News Feed': { type: 'data', inputs: [], outputs: ['news_data'], logic: 'News', function: 'Coleta', dataType: 'text', synergies: ['Text Processing'], incompatibilities: [], desc: 'Notícias' },
            'Market Data': { type: 'data', inputs: [], outputs: ['market_data'], logic: 'Market', function: 'Coleta', dataType: 'timeseries', synergies: ['Technical Analysis'], incompatibilities: [], desc: 'Mercado' },
            'Weather API': { type: 'data', inputs: [], outputs: ['weather_data'], logic: 'Weather', function: 'Coleta', dataType: 'timeseries', synergies: ['Feature Engineering'], incompatibilities: [], desc: 'Clima' },
            'Geospatial Data': { type: 'data', inputs: [], outputs: ['geo_data'], logic: 'Geo', function: 'Leitura', dataType: 'geospatial', synergies: ['Geospatial Processing'], incompatibilities: [], desc: 'Geoespacial' },
            'Graph Data': { type: 'data', inputs: [], outputs: ['graph_data'], logic: 'Graph', function: 'Leitura', dataType: 'graph', synergies: ['Graph Neural Network'], incompatibilities: [], desc: 'Grafo' },
            'Network Data': { type: 'data', inputs: [], outputs: ['network_data'], logic: 'Network', function: 'Leitura', dataType: 'network', synergies: ['Network Analysis'], incompatibilities: [], desc: 'Rede' },
            'Log Files': { type: 'data', inputs: [], outputs: ['log_data'], logic: 'Logs', function: 'Leitura', dataType: 'text', synergies: ['Log Analysis'], incompatibilities: [], desc: 'Logs' },
            'Database Backup': { type: 'data', inputs: [], outputs: ['backup_data'], logic: 'Backup', function: 'Leitura', dataType: 'structured', synergies: ['Restoration'], incompatibilities: [], desc: 'Backup' },
            'Data Lake': { type: 'data', inputs: [], outputs: ['datalake_data'], logic: 'DataLake', function: 'Leitura', dataType: 'bigdata', synergies: ['Data Warehouse'], incompatibilities: [], desc: 'Data Lake' },
            'Data Warehouse': { type: 'data', inputs: [], outputs: ['warehouse_data'], logic: 'Warehouse', function: 'Leitura', dataType: 'structured', synergies: ['Aggregation'], incompatibilities: [], desc: 'Warehouse' },
            'Federated Learning': { type: 'data', inputs: [], outputs: ['federated_data'], logic: 'Federated', function: 'Coleta', dataType: 'distributed', synergies: ['Distributed Training'], incompatibilities: [], desc: 'Federado' },
            'Blockchain Data': { type: 'data', inputs: [], outputs: ['blockchain_data'], logic: 'Blockchain', function: 'Leitura', dataType: 'distributed', synergies: ['Anomaly Detection'], incompatibilities: [], desc: 'Blockchain' },
            'Medical Records': { type: 'data', inputs: [], outputs: ['medical_data'], logic: 'Medical', function: 'Leitura', dataType: 'structured', synergies: ['Data Privacy'], incompatibilities: [], desc: 'Médico' },
            'Genomic Data': { type: 'data', inputs: [], outputs: ['genomic_data'], logic: 'Genomic', function: 'Leitura', dataType: 'sequence', synergies: ['Sequence Analysis'], incompatibilities: [], desc: 'Genômico' },
            'Satellite Imagery': { type: 'data', inputs: [], outputs: ['satellite_data'], logic: 'Satellite', function: 'Leitura', dataType: 'image', synergies: ['Image Processing'], incompatibilities: [], desc: 'Satélite' },
            'LIDAR Data': { type: 'data', inputs: [], outputs: ['lidar_data'], logic: 'LIDAR', function: 'Leitura', dataType: '3d', synergies: ['3D Processing'], incompatibilities: [], desc: 'LIDAR' },
            'Point Cloud': { type: 'data', inputs: [], outputs: ['pointcloud_data'], logic: 'PointCloud', function: 'Leitura', dataType: '3d', synergies: ['3D Processing'], incompatibilities: [], desc: 'Nuvem 3D' },
            'Simulation Data': { type: 'data', inputs: [], outputs: ['simulation_data'], logic: 'Simulation', function: 'Leitura', dataType: 'synthetic', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Simulação' },
            'Experiment Data': { type: 'data', inputs: [], outputs: ['experiment_data'], logic: 'Experiment', function: 'Leitura', dataType: 'structured', synergies: ['Statistical Analysis'], incompatibilities: [], desc: 'Experimento' },
            'Survey Data': { type: 'data', inputs: [], outputs: ['survey_data'], logic: 'Survey', function: 'Leitura', dataType: 'categorical', synergies: ['Categorical Encoding'], incompatibilities: [], desc: 'Pesquisa' },
            'User Behavior': { type: 'data', inputs: [], outputs: ['behavior_data'], logic: 'Behavior', function: 'Coleta', dataType: 'event', synergies: ['Event Processing'], incompatibilities: [], desc: 'Comportamento' },
            'Click Stream': { type: 'data', inputs: [], outputs: ['clickstream_data'], logic: 'ClickStream', function: 'Coleta', dataType: 'event', synergies: ['Event Processing'], incompatibilities: [], desc: 'Clicks' },
            'Transaction Data': { type: 'data', inputs: [], outputs: ['transaction_data'], logic: 'Transaction', function: 'Leitura', dataType: 'structured', synergies: ['Fraud Detection'], incompatibilities: [], desc: 'Transações' },
            'Recommendation Data': { type: 'data', inputs: [], outputs: ['recommendation_data'], logic: 'Recommendation', function: 'Leitura', dataType: 'structured', synergies: ['Collaborative Filtering'], incompatibilities: [], desc: 'Recomendação' },
            'Customer Data': { type: 'data', inputs: [], outputs: ['customer_data'], logic: 'Customer', function: 'Leitura', dataType: 'structured', synergies: ['Customer Segmentation'], incompatibilities: [], desc: 'Clientes' },
            'Product Data': { type: 'data', inputs: [], outputs: ['product_data'], logic: 'Product', function: 'Leitura', dataType: 'structured', synergies: ['Product Analysis'], incompatibilities: [], desc: 'Produtos' },
            'Inventory Data': { type: 'data', inputs: [], outputs: ['inventory_data'], logic: 'Inventory', function: 'Leitura', dataType: 'structured', synergies: ['Demand Forecasting'], incompatibilities: [], desc: 'Inventário' },
            'Supply Chain': { type: 'data', inputs: [], outputs: ['supply_data'], logic: 'Supply', function: 'Leitura', dataType: 'structured', synergies: ['Optimization'], incompatibilities: [], desc: 'Supply Chain' },
            'HR Data': { type: 'data', inputs: [], outputs: ['hr_data'], logic: 'HR', function: 'Leitura', dataType: 'structured', synergies: ['Employee Analytics'], incompatibilities: [], desc: 'RH' },
            'Financial Data': { type: 'data', inputs: [], outputs: ['financial_data'], logic: 'Financial', function: 'Leitura', dataType: 'structured', synergies: ['Financial Analysis'], incompatibilities: [], desc: 'Financeiro' },

            // PRÉ-PROCESSAMENTO (60+ blocos)
            'Normalization': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['normalized_data'], logic: 'Min-Max', function: 'Transformação', dataType: 'normalized', synergies: ['Linear Regression', 'Neural Network'], incompatibilities: [], desc: 'Normaliza 0-1' },
            'Standardization': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['standardized_data'], logic: 'Z-score', function: 'Transformação', dataType: 'standardized', synergies: ['SVM', 'KNN'], incompatibilities: [], desc: 'Padroniza Z-score' },
            'Min-Max Scaling': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['scaled_data'], logic: 'MinMax', function: 'Transformação', dataType: 'scaled', synergies: ['Neural Network'], incompatibilities: [], desc: 'Escala MinMax' },
            'Log Transformation': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['log_transformed'], logic: 'Log', function: 'Transformação', dataType: 'transformed', synergies: ['Linear Regression'], incompatibilities: [], desc: 'Log Transform' },
            'Box-Cox Transform': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['boxcox_data'], logic: 'BoxCox', function: 'Transformação', dataType: 'transformed', synergies: ['Linear Regression'], incompatibilities: [], desc: 'Box-Cox' },
            'Yeo-Johnson Transform': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['yeojohnson_data'], logic: 'YeoJohnson', function: 'Transformação', dataType: 'transformed', synergies: ['Linear Regression'], incompatibilities: [], desc: 'Yeo-Johnson' },
            'Robust Scaling': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['robust_scaled'], logic: 'Robust', function: 'Transformação', dataType: 'scaled', synergies: ['SVM', 'Outlier Detection'], incompatibilities: [], desc: 'Robust Scale' },
            'Quantile Transform': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['quantile_data'], logic: 'Quantile', function: 'Transformação', dataType: 'transformed', synergies: ['Neural Network'], incompatibilities: [], desc: 'Quantile' },
            'Power Transform': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['power_transformed'], logic: 'Power', function: 'Transformação', dataType: 'transformed', synergies: ['Linear Regression'], incompatibilities: [], desc: 'Power' },
            'Data Cleaning': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['clean_data'], logic: 'Clean', function: 'Limpeza', dataType: 'clean', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Limpeza' },
            'Outlier Detection': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['outlier_flagged'], logic: 'Outlier', function: 'Detecção', dataType: 'flagged', synergies: ['Robust Scaling'], incompatibilities: [], desc: 'Outliers' },
            'Outlier Removal': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['clean_data'], logic: 'RemoveOut', function: 'Limpeza', dataType: 'clean', synergies: ['Normalization'], incompatibilities: [], desc: 'Remove Outliers' },
            'IQR Method': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['iqr_cleaned'], logic: 'IQR', function: 'Limpeza', dataType: 'clean', synergies: ['Normalization'], incompatibilities: [], desc: 'IQR' },
            'Z-Score Filtering': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['zscore_cleaned'], logic: 'ZScore', function: 'Limpeza', dataType: 'clean', synergies: ['Standardization'], incompatibilities: [], desc: 'Z-Score' },
            'Isolation Forest': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['if_cleaned'], logic: 'IsoForest', function: 'Detecção', dataType: 'flagged', synergies: ['Anomaly Detection'], incompatibilities: [], desc: 'Isolation Forest' },
            'Local Outlier Factor': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['lof_cleaned'], logic: 'LOF', function: 'Detecção', dataType: 'flagged', synergies: ['Clustering'], incompatibilities: [], desc: 'LOF' },
            'Missing Value Imputation': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['imputed_data'], logic: 'Impute', function: 'Imputação', dataType: 'imputed', synergies: ['Cleaning'], incompatibilities: [], desc: 'Imputação' },
            'Mean Imputation': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['imputed_data'], logic: 'Mean', function: 'Imputação', dataType: 'imputed', synergies: ['Normalization'], incompatibilities: [], desc: 'Média' },
            'Median Imputation': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['imputed_data'], logic: 'Median', function: 'Imputação', dataType: 'imputed', synergies: ['Robust Scaling'], incompatibilities: [], desc: 'Mediana' },
            'Forward Fill': { type: 'preprocessing', inputs: ['timeseries_data'], outputs: ['filled_data'], logic: 'FFill', function: 'Imputação', dataType: 'timeseries', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Forward Fill' },
            'Backward Fill': { type: 'preprocessing', inputs: ['timeseries_data'], outputs: ['filled_data'], logic: 'BFill', function: 'Imputação', dataType: 'timeseries', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Backward Fill' },
            'Interpolation': { type: 'preprocessing', inputs: ['timeseries_data'], outputs: ['interpolated_data'], logic: 'Interp', function: 'Imputação', dataType: 'timeseries', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Interpolação' },
            'KNN Imputation': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['imputed_data'], logic: 'KNN', function: 'Imputação', dataType: 'imputed', synergies: ['Feature Selection'], incompatibilities: [], desc: 'KNN Impute' },
            'MICE Imputation': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['imputed_data'], logic: 'MICE', function: 'Imputação', dataType: 'imputed', synergies: ['Feature Selection'], incompatibilities: [], desc: 'MICE' },
            'Feature Selection': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['selected_features'], logic: 'Select', function: 'Redução', dataType: 'features', synergies: ['Linear Regression', 'Random Forest'], incompatibilities: [], desc: 'Seleção Features' },
            'Variance Threshold': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['selected_features'], logic: 'Variance', function: 'Redução', dataType: 'features', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Variância' },
            'Correlation Analysis': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['correlation_matrix'], logic: 'Corr', function: 'Análise', dataType: 'analysis', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Correlação' },
            'Multicollinearity Detection': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['vif_scores'], logic: 'VIF', function: 'Detecção', dataType: 'analysis', synergies: ['Feature Selection'], incompatibilities: [], desc: 'VIF' },
            'Mutual Information': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['mi_scores'], logic: 'MI', function: 'Seleção', dataType: 'scores', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Informação Mútua' },
            'Chi-Square Test': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['chi2_scores'], logic: 'Chi2', function: 'Teste', dataType: 'scores', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Chi-Square' },
            'ANOVA F-Test': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['f_scores'], logic: 'ANOVA', function: 'Teste', dataType: 'scores', synergies: ['Feature Selection'], incompatibilities: [], desc: 'ANOVA' },
            'Recursive Feature Elimination': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['rfe_features'], logic: 'RFE', function: 'Seleção', dataType: 'features', synergies: ['Feature Selection'], incompatibilities: [], desc: 'RFE' },
            'Permutation Importance': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['importance_scores'], logic: 'Perm', function: 'Análise', dataType: 'scores', synergies: ['Feature Selection'], incompatibilities: [], desc: 'Permutação' },
            'SHAP Values': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['shap_values'], logic: 'SHAP', function: 'Interpretação', dataType: 'explanation', synergies: ['Feature Selection'], incompatibilities: [], desc: 'SHAP' },
            'PCA': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['pca_components'], logic: 'PCA', function: 'Redução', dataType: 'components', synergies: ['Visualization', 'Clustering'], incompatibilities: [], desc: 'PCA' },
            't-SNE': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['tsne_embedding'], logic: 'tSNE', function: 'Redução', dataType: 'embedding', synergies: ['Visualization'], incompatibilities: [], desc: 't-SNE' },
            'UMAP': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['umap_embedding'], logic: 'UMAP', function: 'Redução', dataType: 'embedding', synergies: ['Visualization'], incompatibilities: [], desc: 'UMAP' },
            'Autoencoder': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['encoded_features'], logic: 'AE', function: 'Redução', dataType: 'encoded', synergies: ['Neural Network'], incompatibilities: [], desc: 'Autoencoder' },
            'Tokenization': { type: 'preprocessing', inputs: ['text_data'], outputs: ['tokens'], logic: 'Token', function: 'Processamento', dataType: 'tokens', synergies: ['NLP', 'Text Processing'], incompatibilities: [], desc: 'Tokenização' },
            'Lemmatization': { type: 'preprocessing', inputs: ['text_data'], outputs: ['lemmatized_text'], logic: 'Lemma', function: 'Processamento', dataType: 'text', synergies: ['NLP'], incompatibilities: [], desc: 'Lematização' },
            'Stemming': { type: 'preprocessing', inputs: ['text_data'], outputs: ['stemmed_text'], logic: 'Stem', function: 'Processamento', dataType: 'text', synergies: ['NLP'], incompatibilities: [], desc: 'Stemming' },
            'Stop Words Removal': { type: 'preprocessing', inputs: ['text_data'], outputs: ['filtered_text'], logic: 'StopWords', function: 'Processamento', dataType: 'text', synergies: ['NLP'], incompatibilities: [], desc: 'Stop Words' },
            'TF-IDF Vectorization': { type: 'preprocessing', inputs: ['text_data'], outputs: ['tfidf_vectors'], logic: 'TFIDF', function: 'Vetorização', dataType: 'vectors', synergies: ['Text Classification'], incompatibilities: [], desc: 'TF-IDF' },
            'Word2Vec': { type: 'preprocessing', inputs: ['text_data'], outputs: ['word2vec_embeddings'], logic: 'W2V', function: 'Embedding', dataType: 'embeddings', synergies: ['NLP'], incompatibilities: [], desc: 'Word2Vec' },
            'GloVe': { type: 'preprocessing', inputs: ['text_data'], outputs: ['glove_embeddings'], logic: 'GloVe', function: 'Embedding', dataType: 'embeddings', synergies: ['NLP'], incompatibilities: [], desc: 'GloVe' },
            'FastText': { type: 'preprocessing', inputs: ['text_data'], outputs: ['fasttext_embeddings'], logic: 'FastText', function: 'Embedding', dataType: 'embeddings', synergies: ['NLP'], incompatibilities: [], desc: 'FastText' },
            'Image Resizing': { type: 'preprocessing', inputs: ['image_data'], outputs: ['resized_images'], logic: 'Resize', function: 'Processamento', dataType: 'image', synergies: ['CNN'], incompatibilities: [], desc: 'Redimensionamento' },
            'Image Normalization': { type: 'preprocessing', inputs: ['image_data'], outputs: ['normalized_images'], logic: 'NormImg', function: 'Processamento', dataType: 'image', synergies: ['CNN'], incompatibilities: [], desc: 'Normalização Imagem' },
            'Image Augmentation': { type: 'preprocessing', inputs: ['image_data'], outputs: ['augmented_images'], logic: 'Augment', function: 'Augmentação', dataType: 'image', synergies: ['CNN'], incompatibilities: [], desc: 'Augmentação' },
            'Grayscale Conversion': { type: 'preprocessing', inputs: ['image_data'], outputs: ['grayscale_images'], logic: 'Gray', function: 'Processamento', dataType: 'image', synergies: ['CNN'], incompatibilities: [], desc: 'Escala Cinza' },
            'Edge Detection': { type: 'preprocessing', inputs: ['image_data'], outputs: ['edge_images'], logic: 'Edge', function: 'Processamento', dataType: 'image', synergies: ['CNN'], incompatibilities: [], desc: 'Detecção Bordas' },
            'Denoising': { type: 'preprocessing', inputs: ['image_data'], outputs: ['denoised_data'], logic: 'Denoise', function: 'Limpeza', dataType: 'clean', synergies: ['Image Processing'], incompatibilities: [], desc: 'Denoising' },
            'Detrending': { type: 'preprocessing', inputs: ['timeseries_data'], outputs: ['detrended_data'], logic: 'Detrend', function: 'Processamento', dataType: 'timeseries', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Detrending' },
            'Differencing': { type: 'preprocessing', inputs: ['timeseries_data'], outputs: ['differenced_data'], logic: 'Diff', function: 'Processamento', dataType: 'timeseries', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Diferenciação' },
            'Seasonal Decomposition': { type: 'preprocessing', inputs: ['timeseries_data'], outputs: ['decomposed_components'], logic: 'Seasonal', function: 'Análise', dataType: 'components', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Decomposição' },
            'Lag Features': { type: 'preprocessing', inputs: ['timeseries_data'], outputs: ['lag_features'], logic: 'Lag', function: 'Feature Engineering', dataType: 'features', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Lag Features' },
            'Rolling Window': { type: 'preprocessing', inputs: ['timeseries_data'], outputs: ['rolling_features'], logic: 'Rolling', function: 'Feature Engineering', dataType: 'features', synergies: ['Time Series Analysis'], incompatibilities: [], desc: 'Janela Móvel' },
            'One-Hot Encoding': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['encoded_data'], logic: 'OneHot', function: 'Codificação', dataType: 'encoded', synergies: ['Neural Network'], incompatibilities: [], desc: 'One-Hot' },
            'Label Encoding': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['encoded_data'], logic: 'Label', function: 'Codificação', dataType: 'encoded', synergies: ['Tree Models'], incompatibilities: [], desc: 'Label Encoding' },
            'Target Encoding': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['encoded_data'], logic: 'Target', function: 'Codificação', dataType: 'encoded', synergies: ['Gradient Boosting'], incompatibilities: [], desc: 'Target Encoding' },
            'Ordinal Encoding': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['encoded_data'], logic: 'Ordinal', function: 'Codificação', dataType: 'encoded', synergies: ['Tree Models'], incompatibilities: [], desc: 'Ordinal' },
            'Binary Encoding': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['encoded_data'], logic: 'Binary', function: 'Codificação', dataType: 'encoded', synergies: ['Neural Network'], incompatibilities: [], desc: 'Binary' },
            'Hashing Vectorizer': { type: 'preprocessing', inputs: ['text_data'], outputs: ['hash_vectors'], logic: 'Hash', function: 'Vetorização', dataType: 'vectors', synergies: ['Text Classification'], incompatibilities: [], desc: 'Hash Vectorizer' },
            'Count Vectorizer': { type: 'preprocessing', inputs: ['text_data'], outputs: ['count_vectors'], logic: 'Count', function: 'Vetorização', dataType: 'vectors', synergies: ['Text Classification'], incompatibilities: [], desc: 'Count Vectorizer' },
            'Polynomial Features': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['polynomial_features'], logic: 'Poly', function: 'Feature Engineering', dataType: 'features', synergies: ['Linear Regression'], incompatibilities: [], desc: 'Polinômios' },
            'Interaction Features': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['interaction_features'], logic: 'Interact', function: 'Feature Engineering', dataType: 'features', synergies: ['Linear Regression'], incompatibilities: [], desc: 'Interações' },
            'Binning': { type: 'preprocessing', inputs: ['tabular_data'], outputs: ['binned_data'], logic: 'Bin', function: 'Transformação', dataType: 'categorical', synergies: ['Tree Models'], incompatibilities: [], desc: 'Binning' },
'Feature Engineering': {
  type: 'preprocessing',
  inputs: ['tabular_data'],
  outputs: ['engineered_features'],
  logic: 'Feature_Creation',
  function: 'Transformação',
  synergies: ['Polynomial Features', 'Interaction Features'],
  incompatibilities: [],
  desc: 'Cria novas features por combinação/interação'
},
'XGBoost Classifier': {
  type: 'model',
  inputs: ['clean_data'],
  outputs: ['classification_predictions', 'feature_importance'],
  logic: 'Gradient_Boosting',
  function: 'Classificação',
  synergies: ['Feature Selection', 'Data Cleaning'],
  incompatibilities: ['Linear Regression'],
  desc: 'XGBoost para classificação binária/multiclasse'
},
'Cross Validation': {
  type: 'logic',
  inputs: ['model_object', 'tabular_data'],
  outputs: ['cv_scores', 'stability_report'],
  logic: 'K_Fold',
  function: 'Validação',
  synergies: ['Auto-Model Selector', 'Grid Search'],
  incompatibilities: [],
  desc: 'Validação cruzada K‑Fold para avaliação robusta'
},

'Hyperparameter Tuner': {
  type: 'logic',
  inputs: ['model_object', 'tabular_data'],
  outputs: ['optimized_model', 'best_params'],
  logic: 'Grid_Search',
  function: 'Otimização',
  synergies: ['Cross Validation'],
  incompatibilities: [],
  desc: 'Busca em grade pelos melhores hiperparâmetros'
},
'Matplotlib Chart': {
  type: 'output',
  inputs: ['visualization_data'],
  outputs: [],
  logic: 'Chart',
  function: 'Visualização',
  synergies: ['DataFrame Export'],
  incompatibilities: [],
  desc: 'Gera gráficos de linha, barra, dispersão etc.'
},

'Database Insert': {
  type: 'output',
  inputs: ['predictions_list'],
  outputs: [],
  logic: 'SQL_Insert',
  function: 'Armazenamento',
  synergies: ['Data Warehouse'],
  incompatibilities: [],
  desc: 'Insere resultados em tabela SQL'
},

'Email Report': {
  type: 'output',
  inputs: ['report_text', 'attachment_data'],
  outputs: [],
  logic: 'SMTP',
  function: 'Notificação',
  synergies: ['Scheduled Trigger'],
  incompatibilities: [],
  desc: 'Envia relatório por e‑mail com anexo'
},

'REST API Endpoint': {
  type: 'output',
  inputs: ['predictions_list'],
  outputs: [],
  logic: 'HTTP_Server',
  function: 'Serviço',
  synergies: ['Model Deployment'],
  incompatibilities: [],
  desc: 'Disponibiliza resultados via API REST'
},
'Database SQL': {
  type: 'data',
  inputs: [],
  outputs: ['sql_data'],
  logic: 'SQL_Query',
  function: 'Leitura',
  synergies: ['Data Cleaning', 'Feature Engineering'],
  incompatibilities: [],
  desc: 'Consulta a bancos SQL (MySQL, PostgreSQL, SQLite)'
},
'Blockchain Data': {
  type: 'data',
  inputs: [],
  outputs: ['blockchain_data'],
  logic: 'Blockchain',
  function: 'Leitura',
  synergies: ['Anomaly Detection', 'Graph Neural Network', 'Fraud Detection'],
  incompatibilities: [],
  desc: 'Dados on‑chain: transações, blocos, eventos de contratos inteligentes'
},
'IoT Sensor Data': {
  type: 'data',
  inputs: [],
  outputs: ['sensor_stream'],
  logic: 'MQTT',
  function: 'Coleta',
  synergies: ['Data Cleaning', 'Anomaly Detection', 'Real-time Processing'],
  incompatibilities: [],
  desc: 'Dados de sensores IoT (temperatura, umidade, pressão, movimento) via MQTT'
},

'PDF Invoice Parser': {
  type: 'data',
  inputs: [],
  outputs: ['structured_invoice_data'],
  logic: 'OCR',
  function: 'Extração',
  synergies: ['Text Processing', 'Data Cleaning', 'Fraud Detection'],
  incompatibilities: [],
  desc: 'Extrai campos estruturados de faturas/notas fiscais em PDF (OCR)'
},
'LLM Reversa': { 
    type: 'model', 
    inputs: ['verified_facts', 'address_pointer'], 
    outputs: ['engine_calculation'], 
    logic: 'Determinística', 
    function: 'Cálculo Puro', 
    synergies: ['Blockchain de Fatos', 'Regressão de Endereçamento', 'Sovereign Cache (.cache)', 'Projeção Galvânica', 'Transformer'], 
    desc: 'Paradigma Reverso: o modelo não armazena dados, ele apenas opera cálculos matemáticos sobre um Ledger externo de fatos.' 
},

'Blockchain de Fatos': { 
    type: 'data', 
    inputs: ['raw_data'], 
    outputs: ['verified_facts'], 
    logic: 'SHA-256 / Hash', 
    function: 'Imutabilidade', 
    synergies: ['LLM Reversa', 'Sovereign Cache (.cache)', 'Data Sanitizer', 'Data Schema Validator', 'Fraud Detection'], 
    desc: 'Ledger imutável que sela nexos lógicos em blocos. Elimina alucinações garantindo que cada resposta tenha um lastro verificado.' 
},

'Micro-Sharding Quântico': { 
    type: 'preprocessing', 
    inputs: ['raw_text'], 
    outputs: ['atomic_tokens'], 
    logic: 'Fragmentação Atômica', 
    function: 'Extração de DNA', 
    synergies: ['DFT Linguística', 'Subrepção (The Squeeze)', 'Word2Vec', 'Tokenization', 'BERT Embeddings1'], 
    desc: 'Fragmenta dados em unidades microscópicas (4-8 tokens). Captura a intenção da ação antes mesmo da formação da gramática.' 
},

'Sovereign Cache (.cache)': { 
    type: 'logic', 
    inputs: ['engine_calculation'], 
    outputs: ['memory_pointer'], 
    logic: 'Memória Externa', 
    function: 'Escalabilidade', 
    synergies: ['Blockchain de Fatos', 'LLM Reversa', 'Redis Cache Input', 'Database SQL', 'Model Registry'], 
    desc: 'Separação física entre Cérebro e Memória. Permite que o modelo aprenda infinitamente sem precisar de re-treinamento de pesos.' 
},
'Subrepção (The Squeeze)': { 
    type: 'preprocessing', 
    inputs: ['atomic_tokens'], 
    outputs: ['pure_variables'], 
    logic: 'Compressão de Ruído', 
    function: 'Filtro Puro', 
    synergies: ['Loop de 3 (Triple Loop)', 'Normalization', 'Missing Value Imputation', 'Feature Selection'], 
    desc: 'Técnica de supressão de ruído textual através de janelas decrescentes. Força o modelo a focar apenas na variável que altera o resultado.' 
},

'Loop de 3 (Triple Loop)': { 
    type: 'logic', 
    inputs: ['pure_variables'], 
    outputs: ['stable_nexo'], 
    logic: 'Maturação Híbrida', 
    function: 'Estabilização', 
    synergies: ['Subrepção (The Squeeze)', 'Reinforcement Learning (RL)', 'Cross Validation', 'Hyperparameter Tuner'], 
    desc: 'Ciclo triplo de amadurecimento lógico. Refina o nexo entre os dados três vezes com taxas de aprendizado decrescentes para eliminar a entropia.' 
},

'Projeção Galvânica': { 
    type: 'model', 
    inputs: ['pure_variables', 'stable_nexo'], 
    outputs: ['galvanic_vectors'], 
    logic: 'Transformer Linear', 
    function: 'Ultra-Latência', 
    synergies: ['Transformer', 'Parallel Processor', 'LLM Reversa', 'Linear Weighted Sum', 'Sigmoid Function'], 
    desc: 'Substitui a lentidão do Softmax por Projeção Ortogonal. Reduz o processamento de milissegundos para microsegundos.' 
},
'DFT Linguística': { 
    type: 'logic', 
    inputs: ['atomic_tokens'], 
    outputs: ['entropy_mass'], 
    logic: 'Densidade Funcional', 
    function: 'Cálculo de Energia', 
    synergies: ['Vetor de Ethos Dinâmico', 'Standard Deviation (Z-Score)', 'Sentiment Analysis', 'Gaussian PDF (Normal Distribution)'], 
    desc: 'Trata a linguagem como física de partículas. Calcula a energia da frase pela raridade das palavras e proximidade vetorial.' 
},

'Hierarquia de 7 Camadas': { 
    type: 'logic', 
    inputs: ['stable_nexo'], 
    outputs: ['contextual_scaffold'], 
    logic: 'Comando Triangular', 
    function: 'Orquestração', 
    synergies: ['Action Triad Mapper', 'Decision Tree', 'Auto-Model Selector', 'Consensus Engine (The Judge)'], 
    desc: 'Estrutura de comando que organiza o fluxo do Macro (Lenda) ao Micro (Átomo), impedindo que detalhes quebrem o contexto geral.' 
},

'Action Triad Mapper': { 
    type: 'preprocessing', 
    inputs: ['contextual_scaffold'], 
    outputs: ['behavioral_tags'], 
    logic: 'Ação/Relação/Reação', 
    function: 'Mapeamento Humano', 
    synergies: ['Hierarquia de 7 Camadas', 'Sinais de Ativação Soberanos', 'NLP Pro', 'Pattern Recognition'], 
    desc: 'Simula o processamento humano de fatos através de tags lógicas de Introdução, Ponte e Conclusão.' 
},
'Vetor de Ethos Dinâmico': { 
    type: 'output', 
    inputs: ['entropy_mass', 'engine_calculation'], 
    outputs: ['calibrated_voice'], 
    logic: 'Calibração de Tom', 
    function: 'Síntese de Voz', 
    synergies: ['DFT Linguística', 'LLM Sentiment Analysis', 'Platt Scaling (Calibration)', 'AI Strategic Analyst (The Voice)'], 
    desc: 'Ajusta o tom de voz da IA em tempo real. Se o nexo for instável, a síntese reflete incerteza; se sólido, reflete autoridade.' 
},

'Regressão de Endereçamento': { 
    type: 'model', 
    inputs: ['entropy_mass'], 
    outputs: ['address_pointer'], 
    logic: 'Ponteiro de Entropia', 
    function: 'Busca Instantânea', 
    synergies: ['LLM Reversa', 'Sovereign Cache (.cache)', 'Cosine Similarity', 'Regressão de Endereçamento'], 
    desc: 'Calcula o endereço físico do dado no cache através da entropia do prompt. Ignora a busca semântica para atingir velocidade de hardware.' 
},

'Sinais de Ativação Soberanos': { 
    type: 'logic', 
    inputs: ['behavioral_tags'], 
    outputs: ['algebra_interrupts'], 
    logic: 'Interruptores Lógicos', 
    function: 'Troca de Álgebra', 
    synergies: ['Action Triad Mapper', 'Boolean Gate (AND/OR/NOT)', 'Threshold Trigger', 'Execution Trigger'], 
    desc: 'Conectivos lógicos (<MAS>, <E>, <SE>) que funcionam como chaves físicas, alterando o comportamento do motor de cálculo durante a síntese.' 
},
  'DSP Audio Normalizer': { 
    type:'preprocessing', 
    inputs:['raw_audio_buffer'], 
    outputs:['normalized_audio'], 
    logic:'Sample_Rate_Conversion / Bit_Depth_Fix', 
    function:'Processamento de Sinal', 
    synergies:['Data Cleaning', 'Spectrogram Generator'], 
    desc:'O Ajustador Acústico: Padroniza a taxa de amostragem (ex: 22kHz ou 44kHz) e remove ruídos de fundo para que a IA processe uma voz limpa.' 
  },

  'Mel-Spectrogram Generator': { 
    type:'preprocessing', 
    inputs:['normalized_audio'], 
    outputs:['mel_spectrogram'], 
    logic:'Short-Time_Fourier_Transform (STFT)', 
    function:'Visão Sonora', 
    synergies:['CNN', 'Transformer'], 
    desc:'O Tradutor Visual: Transforma ondas sonoras em imagens de frequência (Espectrogramas). É assim que a IA "enxerga" o som para aprender a falar.' 
  },

  'TTS Sequence Encoder (FastSpeech 2)': { 
    type:'model', 
    inputs:['text_embeddings', 'prosody_params'], 
    outputs:['spectrogram_prediction'], 
    logic:'Transformer_Architecture', 
    function:'Cérebro Vocal', 
    synergies:['BERT Embeddings', 'Transformer'], 
    desc:'O Gerador de Fala: Pega o texto e decide a duração de cada fonema e a melodia da voz, gerando um espectrograma de como a frase deve soar.' 
  },

  'Neural Vocoder (HiFi-GAN)': { 
    type:'output', 
    inputs:['spectrogram_prediction'], 
    outputs:['final_voice_wave'], 
    logic:'Generative_Adversarial_Network (GAN)', 
    function:'Síntese de Áudio', 
    synergies:['TTS Sequence Encoder', 'Neural Network'], 
    desc:'O Músculo Vocal: Transforma o espectrograma matemático em áudio realista de alta fidelidade. É o que dá a "textura" humana à voz do robô.' 
  },

  'Prosody & Pitch Controller': { 
    type:'logic', 
    inputs:['spectrogram_prediction'], 
    outputs:['expressive_spectrogram'], 
    logic:'F0_Contour_Manipulation', 
    function:'Design de Personalidade', 
    synergies:['Sentiment Analysis', 'Neural Vocoder'], 
    desc:'O Entonador: Ajusta o tom (pitch) e a velocidade para dar emoção. Se o sniper lucrar, a voz sobe o tom; se o mercado cair, a voz sobe a seriedade.' 
  },

  'Granular Robotic Modulator': { 
    type:'logic', 
    inputs:['final_voice_wave'], 
    outputs:['robotic_signal'], 
    logic:'Granular_Synthesis', 
    function:'Design de Som', 
    synergies:['Neural Vocoder'], 
    desc:'A Identidade Digital: Adiciona texturas metálicas ou granulares para que a voz soe como um robô de ficção científica (estilo Behemoth), mantendo a clareza.' 
  },

  'Coqui TTS Trainer Interface': { 
    type:'model', 
    inputs:['audio_dataset', 'text_transcript'], 
    outputs:['custom_voice_model'], 
    logic:'PyTorch_Deep_Learning', 
    function:'Criação de Dataset', 
    synergies:['Model Registry', 'GPU Acceleration'], 
    desc:'O Treinador de Voz: Interface para usar o framework Coqui TTS e criar uma identidade vocal única baseada em poucas horas de gravação.' 
  },
'Flow Entrance (Calm)': { 
    type:'logic', 
    inputs:['start_signal'], 
    outputs:['calm_focus_state'], 
    logic:'Ambient_Soft_Pads (80-90 BPM)', 
    function:'Abertura de Foco', 
    synergies:['JSON Input', 'Data Parsing'], 
    desc:'O Início: Clima calmo para carregar dados. SINERGIA: Use enquanto organiza as primeiras entradas do dia. Música: Ambient digital, texturas leves.' 
  },

  'Focus Stabilizer (Steady)': { 
    type:'logic', 
    inputs:['calm_focus_state'], 
    outputs:['constant_energy_state'], 
    logic:'Minimal_Techno (90 BPM)', 
    function:'Estabilização de Foco', 
    synergies:['Data Cleaning', 'Normalization'], 
    desc:'O Ritmo: Foco constante para tarefas repetitivas de limpeza. Música: Ritmo estável, baixo profundo, atmosfera de codificação.' 
  },

  'Productivity Core (Pulse)': { 
    type:'logic', 
    inputs:['constant_energy_state'], 
    outputs:['immersive_work_state'], 
    logic:'Rhythmic_Synths (95-105 BPM)', 
    function:'Trabalho Imersivo', 
    synergies:['Feature Engineering', 'Auto-Model Selector'], 
    desc:'O Fluxo: Quando a inteligência começa a rodar. Música: Eletrônica pulsante, efeitos glitch, foco total na construção de modelos.' 
  },

  'Execution Peak (Cyberpunk)': { 
    type:'logic', 
    inputs:['immersive_work_state'], 
    outputs:['high_performance_state'], 
    logic:'Fast_Arpeggios (110-125 BPM)', 
    function:'Resolução de Complexidade', 
    synergies:['API Pusher (Webhook)', 'Neural Network'], 
    desc:'O Pico: Energia alta para resolver problemas difíceis ou disparar o Sniper. Música: Cyberpunk vibe, batidas fortes, foco agressivo e controlado.' 
  },

  'Analytic Tension (Hacker)': { 
    type:'logic', 
    inputs:['high_performance_state'], 
    outputs:['deep_analysis_state'], 
    logic:'Dark_Ambient_Techno (85-95 BPM)', 
    function:'Mente Profunda / Análise', 
    synergies:['Post-Mortem Analysis', 'Risk Analysis'], 
    desc:'A Suspensão: Estilo "Mr. Robot" para analisar erros ou riscos ocultos. Música: Sinais distorcidos, baixo tenso, clima psicológico de análise.' 
  },

  'Cycle Deceleration (Closing)': { 
    type:'logic', 
    inputs:['deep_analysis_state'], 
    outputs:['system_standby'], 
    logic:'Soft_Piano_Reverb (60-70 BPM)', 
    function:'Fechamento de Ciclo', 
    synergies:['JSON Export', 'Model Serializer'], 
    desc:'O Desligamento: Clima etéreo para salvar o progresso e descansar. Música: Desaceleração, piano suave, fechando as portas do sistema.' 
  },

  'Contextual Mood Sync': { 
    type:'logic', 
    inputs:['system_status', 'pnl_result'], 
    outputs:['audio_prompt_signal'], 
    logic:'Adaptive_Audio_Bridge', 
    function:'Sincronizador de Clima', 
    synergies:['Flow Entrance', 'Cycle Deceleration'], 
    desc:'O Maestro: Bloco automático que escolhe qual dos 6 blocos de clima acima deve tocar baseado no que o sistema está fazendo no momento.' 
  },
'Mood Query Interface': { 
    type:'logic', 
    inputs:['user_input', 'time_of_day'], 
    outputs:['user_needs_signal'], 
    logic:'Conversational_Logic', 
    function:'Pergunta Ativa', 
    synergies:['AI Strategic Analyst (The Voice)', 'Contextual Mood Sync'], 
    desc:'O Concierge: Periodicamente, o sistema te pergunta como você se sente. Ex: "Quer focar ou quer agitar?". Ele transforma sua resposta em um comando para os blocos de música.' 
  },

  'Dopamine Trigger (Animação)': { 
    type:'logic', 
    inputs:['user_needs_signal'], 
    outputs:['upbeat_audio_prompt'], 
    logic:'Upbeat_Uplifting_Logic', 
    function:'Gerador de Ânimo', 
    synergies:['Execution Peak (Cyberpunk)'], 
    desc:'O Animador: Recomendação para quando você está desanimado. Estilo: Nu-Disco, Synthwave Alegre, batidas vibrantes que te fazem querer agir.' 
  },

  'Adrenaline Booster (Agitação)': { 
    type:'logic', 
    inputs:['user_needs_signal', 'snipe_opportunity'], 
    outputs:['intense_audio_prompt'], 
    logic:'Aggressive_Phonk_Techno', 
    function:'Gerador de Agitação', 
    synergies:['Analytic Tension (Hacker)'], 
    desc:'O Agitador: Para momentos de ação pura. Música: Phonk industrial, Hardcore Techno, batidas agressivas para te deixar em "Modo Combate".' 
  },

  'Voice Prompt Motivator': { 
    type:'output', 
    inputs:['user_needs_signal'], 
    outputs:['voice_message'], 
    logic:'Generative_Speech', 
    function:'Incentivo Vocal', 
    synergies:['Coqui TTS Trainer Interface', 'Neural Vocoder'], 
    desc:'O Coach Digital: A voz que você criou te dá frases de incentivo. Ex: "Os dados estão perfeitos, Ronan. Essa é a sua hora. Vamos pra cima!".' 
  },
'Prompt Template Architect': { 
    type:'logic', 
    inputs:['raw_instruction', 'context_data'], 
    outputs:['formatted_prompt'], 
    logic:'Template_Engine', 
    function:'Engenharia de Prompt', 
    synergies:['BERT Embeddings', 'LLM Sentiment Analysis'], 
    desc:'O Estruturador: Cria a "moldura" da pergunta para a IA. Ele organiza como o sistema deve falar com o Quantikus, garantindo que a resposta seja técnica e direta.' 
  },

  'AI Temperature Controller': { 
    type:'logic', 
    inputs:['formatted_prompt'], 
    outputs:['tuned_prompt'], 
    logic:'Softmax_Temperature_Adjustment', 
    function:'Ajuste de Criatividade', 
    synergies:['Quantikus Core', 'Sigmoid Function'], 
    desc:'O Calibrador de Ousadia: Ajusta a "temperatura" da IA. (Baixa: lógica fria e exata / Alta: criatividade e conexões inusitadas). SINERGIA: Use temperatura baixa para Snipes e alta para análises de literatura.' 
  },

  'Context Grounding (RAG)': { 
    type:'logic', 
    inputs:['tuned_prompt', 'database_facts'], 
    outputs:['grounded_prompt'], 
    logic:'Retrieval_Augmented_Generation', 
    function:'Filtro de Verdade', 
    synergies:['Sentiment Vector Store', 'JSON Input'], 
    desc:'O Ancorador: Garante que a IA não invente coisas (alucinação). Ele força o prompt a se basear apenas nos dados reais que entraram no sistema.' 
  },

  'Logic Override (Hot-Patch)': { 
    type:'logic', 
    inputs:['current_logic_flow', 'user_adjustment'], 
    outputs:['updated_logic_flow'], 
    logic:'Dynamic_Code_Injection', 
    function:'Ajuste de Programação', 
    synergies:['Main Event Loop', 'System Mode Controller'], 
    desc:'O Remendo em Tempo Real: Permite mudar uma regra de programação (ex: mudar um cálculo de lucro) com o sistema rodando, sem precisar desligar o Sniper.' 
  },

  'Heuristic Weight Tuner': { 
    type:'logic', 
    inputs:['model_weights'], 
    outputs:['calibrated_weights'], 
    logic:'Manual_Parameter_Scaling', 
    function:'Ajuste de Pesos', 
    synergies:['Reinforcement Learning', 'Polynomial Regression'], 
    desc:'O Ajustador de Importância: Se você percebe que o sistema está dando muita bola para um dado irrelevante, você usa este bloco para diminuir o peso dele manualmente.' 
  },

  'Compiler Optimization Flag': { 
    type:'logic', 
    inputs:['system_code'], 
    outputs:['optimized_executable'], 
    logic:'Numba_JIT / Cython_Compile', 
    function:'Performance de Código', 
    synergies:['GPU Acceleration', 'Parallel Processor'], 
    desc:'O Acelerador de Motor: Ajusta como o Python lê o código. Ele transforma partes lentas em código de máquina ultra-rápido para ganhar milissegundos no Sniper.' 
  },

  'Semantic Output Guardrail': { 
    type:'logic', 
    inputs:['ai_response'], 
    outputs:['safe_response'], 
    logic:'Regex_Content_Filter', 
    function:'Segurança de Saída', 
    synergies:['AI Strategic Analyst (The Voice)', 'JSON Export'], 
    desc:'O Filtro de Resposta: Analisa o que a IA escreveu antes de te mostrar. Ele remove termos confusos ou "linguiça" e deixa apenas a instrução clara de programação ou de aposta.' 
  },
'For-Each Loop (Iterator)': { 
    type:'logic', 
    inputs:['data_collection'], 
    outputs:['single_item', 'loop_index'], 
    logic:'for(let i=0; i<data.length; i++)', 
    function:'Iteração de Dados', 
    synergies:['JSON Input', 'CSV Input'], 
    desc:'O Repetidor: Pega uma lista de dados (ex: 100 partidas) e processa uma por uma, enviando cada item individualmente para os modelos de IA.' 
  },

  'While-Condition Loop': { 
    type:'logic', 
    inputs:['condition_signal'], 
    outputs:['continue_tick'], 
    logic:'while(condition === true)', 
    function:'Loop Condicional', 
    synergies:['Main Event Loop', 'Threshold Trigger'], 
    desc:'O Vigilante: Mantém um processo rodando enquanto uma condição for verdadeira (ex: "Enquanto o mercado estiver aberto, continue analisando").' 
  },

  'If-Else Branch (Conditional)': { 
    type:'logic', 
    inputs:['boolean_signal', 'data_in'], 
    outputs:['true_path', 'false_path'], 
    logic:'if(A) { do X } else { do Y }', 
    function:'Desvio Lógico', 
    synergies:['Value Comparator', 'Safety Execution Gate'], 
    desc:'A Bifurcação: O bloco de decisão mais importante. Se a condição for atendida, o dado vai pelo caminho A; se não, vai pelo caminho B.' 
  },

  'Try-Catch (Error Handler)': { 
    type:'logic', 
    inputs:['process_input'], 
    outputs:['success_output', 'error_signal'], 
    logic:'try { execute } catch (e) { handle }', 
    function:'Tratamento de Exceção', 
    synergies:['API Pusher (Webhook)', 'Live API Connector'], 
    desc:'O Seguro contra Falhas: Tenta executar uma ação (ex: enviar ordem). Se der erro na internet ou na API, ele captura o erro e impede que o programa inteiro trave.' 
  },

  'Switch-Case (Multi-Router)': { 
    type:'logic', 
    inputs:['input_value'], 
    outputs:['case_1', 'case_2', 'case_3', 'default'], 
    logic:'switch(value) { case A, B, C... }', 
    function:'Roteamento Múltiplo', 
    synergies:['Auto-Model Selector', 'Market Weather Station'], 
    desc:'O Distribuidor: Direciona o fluxo para vários caminhos diferentes baseados em um valor (ex: Se clima="Calmo", vai pro Caminho 1; se "Caos", vai pro Caminho 2).' 
  },

  'Async-Await (Wait for Signal)': { 
    type:'logic', 
    inputs:['trigger'], 
    outputs:['resolved_data'], 
    logic:'await Promise.all([tasks])', 
    function:'Sincronização Assíncrona', 
    synergies:['Live API Connector', 'WebSocket Stream'], 
    desc:'O Sincronizador: Faz o sistema esperar por uma resposta externa antes de prosseguir. Essencial para garantir que o modelo só calcule quando os dados da API chegarem.' 
  },

  'Break/Exit Trigger': { 
    type:'logic', 
    inputs:['kill_signal'], 
    outputs:[], 
    logic:'process.exit(0) / break', 
    function:'Interrupção de Fluxo', 
    synergies:['Emergency Kill Switch', 'Cumulative Loss Guard'], 
    desc:'O Ponto Final: Interrompe imediatamente qualquer loop ou processo que esteja rodando. É o comando de "Parar Tudo" definitivo da programação.' 
  },
 'NN Dense Layer (Linear)': { 
    type:'logic', 
    inputs:['vector_in'], 
    outputs:['layer_out'], 
    logic:'Python: nn.Linear(in_features, out_features) + nn.ReLU()', 
    function:'Conexão Neural', 
    synergies:['Neural Network', 'Normalization'], 
    desc:'O Bloco de Construção: Conecta todos os inputs aos neurônios. Código: "output = activation(dot(W, x) + b)". Essencial para Redes Neurais MLP.' 
  },

  'RNN Recurrent Cell': { 
    type:'logic', 
    inputs:['time_step_in', 'hidden_state'], 
    outputs:['sequence_out', 'new_hidden_state'], 
    logic:'Python: nn.LSTMCell(input_size, hidden_size)', 
    function:'Memória de Curto Prazo', 
    synergies:['RNN', 'Time Series Data'], 
    desc:'O Elo Sequencial: Mantém o estado anterior para entender o próximo. Código: "h_t = tanh(W_h * h_t-1 + W_x * x_t)". Ideal para RNN e LSTM.' 
  },

  'Attention Head (Transformer)': { 
    type:'logic', 
    inputs:['query', 'key', 'value'], 
    outputs:['context_vector'], 
    logic:'Python: MultiheadAttention(embed_dim, num_heads)', 
    function:'Foco Seletivo', 
    synergies:['Transformer', 'BERT Embeddings'], 
    desc:'O Núcleo do Transformer: Calcula a importância de cada dado em relação aos outros. Código: "softmax(QK^T / sqrt(d_k))V".' 
  },

  'Backpropagation Optimizer': { 
    type:'logic', 
    inputs:['loss_value', 'model_weights'], 
    outputs:['updated_weights'], 
    logic:'Python: optimizer.step() | loss.backward()', 
    function:'Aprendizado de Erro', 
    synergies:['Reinforcement Learning', 'Cross-Entropy Loss'], 
    desc:'O Motor de Ajuste: O código que realmente faz a IA aprender. Implementa algoritmos como Adam ou SGD para minimizar o erro do Sniper.' 
  },

  'Data Batcher (DataLoader)': { 
    type:'logic', 
    inputs:['dataset_in'], 
    outputs:['mini_batch_out'], 
    logic:'Python: DataLoader(dataset, batch_size=64, shuffle=True)', 
    function:'Alimentação de IA', 
    synergies:['CSV Input', 'JSON Input'], 
    desc:'O Carregador: Organiza os dados em grupos (batches) para que a GPU processe mais rápido. Sem isso, o treinamento da rede neural seria lento e instável.' 
  },

  'Forward Pass (Inference)': { 
    type:'logic', 
    inputs:['input_data', 'trained_model'], 
    outputs:['prediction_output'], 
    logic:'Python: model.eval() | with torch.no_grad(): output = model(x)', 
    function:'Execução de Modelo', 
    synergies:['API Pusher (Webhook)', 'Polynomial Regression'], 
    desc:'O Tiro da IA: O código que coloca o modelo em "Modo de Produção". Ele ignora o aprendizado e foca apenas em gerar a previsão mais rápida possível.' 
  },

  'Softmax Normalization': { 
    type:'logic', 
    inputs:['logits_out'], 
    outputs:['class_probabilities'], 
    logic:'Python: torch.softmax(x, dim=1) | np.exp(x)/sum(np.exp(x))', 
    function:'Cálculo de Probabilidade', 
    synergies:['Neural Network', 'Argmax (Decision Picker)'], 
    desc:'O Conversor de Confiança: Transforma scores brutos da IA em probabilidades de 0 a 100%. Código: "exp(x) / sum(exp(x))".' 
  },

'Data Object Blueprint': { 
    type:'logic', 
    inputs:['raw_fields'], 
    outputs:['structured_object'], 
    logic:'C#: public class Signal { ... } | Python: @dataclass | JS: const signal = { ... }', 
    function:'Estrutura de Dados', 
    synergies:['JSON Input', 'Data Parsing'], 
    desc:'O Molde: Define como um "Sinal de Sniper" é guardado na memória. \nC#: public record SniperSignal(double Prob, string Asset);\nPython: class Signal: prob: float; asset: str' 
  },

  'Async API Pusher': { 
    type:'output', 
    inputs:['json_payload'], 
    outputs:['response_code'], 
    logic:'Python: requests.post | C#: HttpClient.PostAsync | JS: fetch()', 
    function:'Execução de Código', 
    synergies:['API Pusher (Webhook)', 'JSON Export'], 
    desc:'O Disparo Real: O código que envia a ordem para a web. \nPython: requests.post(url, json=data)\nJS: await fetch(url, {method: "POST", body: JSON.stringify(data)})' 
  },

  'Neural Dot Product': { 
    type:'logic', 
    inputs:['weights', 'inputs'], 
    outputs:['activation_sum'], 
    logic:'Python: np.dot(w, i) | C#: MathNet.Numerics | C++: std::inner_product', 
    function:'Matemática de IA', 
    synergies:['NN Dense Layer (Linear)', 'Neural Network'], 
    desc:'O Cálculo do Neurônio: Multiplicação de matrizes em código. \nC++: std::inner_product(w.begin(), w.end(), i.begin(), 0.0);\nPython: weights @ inputs' 
  },

  'Condition Threshold': { 
    type:'logic', 
    inputs:['probability'], 
    outputs:['trigger_signal'], 
    logic:'All: if (prob > threshold) { execute(); }', 
    function:'Gatilho Lógico', 
    synergies:['Threshold Trigger', 'Safety Execution Gate'], 
    desc:'O Filtro de Decisão: O código clássico de "Atirar ou não". \nC#: if (signal.Prob >= 0.95) ExecuteSnipe();\nPython: if prob >= 0.95: send_webhook()' 
  },

  'Batch Iterator (Loop)': { 
    type:'logic', 
    inputs:['dataset'], 
    outputs:['row_item'], 
    logic:'Python: for row in df | C#: foreach(var item in list) | JS: data.map()', 
    function:'Processamento em Massa', 
    synergies:['For-Each Loop (Iterator)', 'CSV Input'], 
    desc:'O Motor de Repetição: Percorre milhares de dados rapidamente. \nC#: foreach (var entry in historicalData) { Process(entry); }\nJS: dataset.forEach(row => analyze(row));' 
  },

  'JSON Serializer': { 
    type:'logic', 
    inputs:['object'], 
    outputs:['json_string'], 
    logic:'JS: JSON.stringify | Python: json.dumps | C#: JsonSerializer.Serialize', 
    function:'Conversão de Dados', 
    synergies:['JSON Export', 'API Pusher (Webhook)'], 
    desc:'O Empacotador: Transforma o resultado da IA em texto para a web. \nC#: string json = JsonSerializer.Serialize(myResult);\nPython: raw_data = json.dumps(model_output)' 
  },

  'Safety Try-Catch Block': { 
    type:'logic', 
    inputs:['risky_process'], 
    outputs:['safe_result', 'error_log'], 
    logic:'C#/JS: try { ... } catch (err) | Python: try: ... except:', 
    function:'Tratamento de Erros', 
    synergies:['Try-Catch (Error Handler)', 'Emergency Kill Switch'], 
    desc:'O Escudo de Código: Impede que o sistema caia se a internet falhar. \nPython: try: sniper.fire() except ConnectionError as e: log(e)\nJS: try { await api.call() } catch (err) { stop_system() }' 
  },
'C# Decision Logic': { 
    type:'logic', 
    inputs:['probability'], 
    outputs:['signal'], 
    logic:'if (prob >= 0.85f) { SendSignal(); }', 
    function:'C# Implementation', 
    synergies:['Safety Execution Gate', 'API Pusher'], 
    desc:'Código C#: "public void Check(float p) { if(p >= 0.85) Execute(); }". Focado em execução rápida e tipagem forte.' 
  },

  'Python AI Logic': { 
    type:'logic', 
    inputs:['data_frame'], 
    outputs:['prediction'], 
    logic:'df["pred"] = model.predict(X)', 
    function:'Python Implementation', 
    synergies:['Neural Network', 'Normalization'], 
    desc:'Código Python: "prediction = [model.predict(x) for x in data]". A linguagem padrão para manipular tensores e modelos de IA.' 
  },

  'C# Async Loop': { 
    type:'logic', 
    inputs:['data_list'], 
    outputs:['processed_item'], 
    logic:'await Task.Run(() => Parallel.ForEach(data, x => { ... }));', 
    function:'C# Implementation', 
    synergies:['Event Loop', 'Parallel Processor'], 
    desc:'Código C#: "foreach (var item in list) { await ProcessAsync(item); }". Ideal para processar múltiplos snipes ao mesmo tempo sem travar a interface.' 
  },

  'Python List Comprehension': { 
    type:'logic', 
    inputs:['raw_list'], 
    outputs:['filtered_list'], 
    logic:'clean_data = [x for x in data if x > threshold]', 
    function:'Python Implementation', 
    synergies:['Data Cleaning', 'Feature Selection'], 
    desc:'Código Python: "results = [x * 1.1 for x in signals if x.is_valid]". Forma ultra-rápida e elegante de filtrar e transformar dados em Python.' 
  },

  'C# Try-Catch Guard': { 
    type:'logic', 
    inputs:['risky_operation'], 
    outputs:['safe_result'], 
    logic:'try { ... } catch (Exception ex) { Log(ex.Message); }', 
    function:'C# Implementation', 
    synergies:['Emergency Kill Switch', 'Try-Catch (Error Handler)'], 
    desc:'Código C#: "try { client.Send(data); } catch (HttpRequestException e) { Handle(e); }". O padrão ouro para segurança de conexão em C#.' 
  },

  'Python Dictionary Mapping': { 
    type:'logic', 
    inputs:['keys', 'values'], 
    outputs:['data_map'], 
    logic:'data_map = dict(zip(keys, values))', 
    function:'Python Implementation', 
    synergies:['JSON Export', 'Data Parsing'], 
    desc:'Código Python: "config = { "id": 1, "val": 0.95 }". A forma mais poderosa de organizar dados estruturados antes de exportar para JSON.' 
  },

  'C# LINQ Filter': { 
    type:'logic', 
    inputs:['collection'], 
    outputs:['filtered_collection'], 
    logic:'var best = data.Where(x => x.Score > 90).OrderByDescending(x => x.Value);', 
    function:'C# Implementation', 
    synergies:['Feature Selection', 'Expected Value (EV+)'], 
    desc:'Código C#: "var signals = list.Where(s => s.Active).ToList();". Usa a biblioteca LINQ para fazer buscas e filtros complexos em milissegundos.' 
  },
'C++ High-Speed Core': { 
    type:'logic', 
    inputs:['raw_tick'], 
    outputs:['fast_signal'], 
    logic:'C++: if(tick->price > limit) { execute_trade(); }', 
    function:'Performance Extrema', 
    synergies:['Event Loop', 'WebSocket Stream'], 
    desc:'O Motor de Corrida: Usa C++ para processar dados em microssegundos. Ex: "auto start = std::chrono::high_resolution_clock::now();". Ideal para HFT (High Frequency Trading).' 
  },

  'Rust Safety Memory': { 
    type:'logic', 
    inputs:['sensitive_data'], 
    outputs:['secure_output'], 
    logic:'Rust: let signal = match result { Ok(v) => v, Err(e) => panic!(e) };', 
    function:'Segurança de Memória', 
    synergies:['Emergency Kill Switch', 'Try-Catch (Error Handler)'], 
    desc:'O Escudo Inquebrável: Usa Rust para garantir que o sistema nunca trave por erro de memória (Memory Leak). Ex: "pub fn check_safety(data: &Data) -> bool { ... }".' 
  },

  'JS/Node Web Engine': { 
    type:'logic', 
    inputs:['api_data'], 
    outputs:['json_response'], 
    logic:'JS: const results = await Promise.all(api_calls.map(fetch));', 
    function:'Integração Web', 
    synergies:['API Pusher (Webhook)', 'Dashboard'], 
    desc:'O Conector Web: Usa JavaScript (Node.js) para gerenciar milhares de conexões de API ao mesmo tempo sem travar. Ex: "socket.on("message", (data) => process(data));".' 
  },

  'SQL Query Master': { 
    type:'data', 
    inputs:['search_params'], 
    outputs:['db_results'], 
    logic:'SQL: SELECT * FROM signals WHERE prob > 0.90 ORDER BY timestamp DESC;', 
    function:'Busca em Banco de Dados', 
    synergies:['Database SQL', 'PostgreSQL/MySQL'], 
    desc:'O Bibliotecário: Extrai dados históricos de bancos gigantes instantaneamente. Ex: "JOIN historical_data ON signals.id = historical_data.id".' 
  },

  'Python/C# Hybrid Logic': { 
    type:'logic', 
    inputs:['model_output', 'ui_config'], 
    outputs:['final_decision'], 
    logic:'Py: result = model.predict() | C#: if(result.Score > ui.Min) { ... }', 
    function:'Lógica Híbrida', 
    synergies:['Auto-Model Selector', 'User UI Input'], 
    desc:'O Cérebro Integrador: Usa Python para a inteligência e C# para a interface de usuário. É a mistura perfeita entre IA e usabilidade profissional.' 
  },

  'Go/Golang Concurrency': { 
    type:'logic', 
    inputs:['task_list'], 
    outputs:['completed_tasks'], 
    logic:'Go: go processTask(item) // goroutine', 
    function:'Processamento Paralelo', 
    synergies:['Parallel Processor', 'Batch Iterator (Loop)'], 
    desc:'O Multi-Tarefa: Usa a linguagem Go para rodar centenas de modelos de IA em paralelo usando "Goroutines", que são muito mais leves que processos comuns.' 
  },

  'Bash/Shell Automation': { 
    type:'output', 
    inputs:['system_status'], 
    outputs:['auto_restart'], 
    logic:'Bash: if [ $status -eq 0 ]; then systemctl restart cardus; fi', 
    function:'Automação de Servidor', 
    synergies:['Heartbeat Monitor', 'Emergency Kill Switch'], 
    desc:'O Zelador: Scripts de Linux para garantir que, se o sistema cair, ele se reinicie sozinho em 1 segundo. Ex: "while true; do ./cardus_sniper; done".' 
  },
'AI Model Dispatcher': { 
    type:'logic', 
    inputs:['raw_features'], 
    outputs:['selected_model_path'], 
    logic:'def route(data): return "RNN" if data.is_sequence else "XGBoost"', 
    function:'Roteamento de Dados', 
    synergies:['Auto-Model Selector', 'RNN', 'XGBoost'], 
    desc:'O Despachante: Uma função que analisa o dado e decide qual modelo terá maior precisão. Ex: Se o dado for uma série temporal, ele despacha para a RNN; se for tabular, para o XGBoost.' 
  },

  'Feature Importance Evaluator': { 
    type:'logic', 
    inputs:['model_output'], 
    outputs:['top_features_list'], 
    logic:'def evaluate(m): return m.feature_importances_', 
    function:'Análise de Relevância', 
    synergies:['Feature Selection', 'Random Forest', 'SHAP Explainer'], 
    desc:'O Crítico: Uma função que pergunta ao modelo: "O que você usou para tomar essa decisão?". Ela filtra as variáveis inúteis (linguiça) e foca no que realmente deu o sinal.' 
  },

  'Confidence Score Validator': { 
    type:'logic', 
    inputs:['prediction_prob'], 
    outputs:['validated_signal'], 
    logic:'def check(p): return p if p > 0.92 else None', 
    function:'Validação de Certeza', 
    synergies:['Sigmoid Function', 'Safety Execution Gate', 'Platt Scaling'], 
    desc:'O Juiz de Confiança: Uma função que barra predições "mornas". Ela só deixa o sinal passar se a probabilidade da IA for superior a um limite rigoroso (ex: 92%).' 
  },

  'Ensemble Weight Blender': { 
    type:'logic', 
    inputs:['pred_a', 'pred_b', 'pred_c'], 
    outputs:['weighted_final_pred'], 
    logic:'def blend(a, b, c): return (a*0.5 + b*0.3 + c*0.2)', 
    function:'Fusão de Modelos', 
    synergies:['Linear Weighted Sum', 'Neural Network', 'Consensus Engine'], 
    desc:'O Liquidificador de IA: Uma função que combina o resultado de vários modelos. Ela dá mais peso para o modelo mais confiável e gera uma resposta única e robusta.' 
  },

  'Data Pipeline Orchestrator': { 
    type:'logic', 
    inputs:['raw_input'], 
    outputs:['processed_tensor'], 
    logic:'def pipe(x): return scale(clean(parse(x)))', 
    function:'Automação de Fluxo', 
    synergies:['Data Parsing', 'Data Cleaning', 'Normalization'], 
    desc:'O Maestro do Fluxo: Uma função que garante que o dado passe pela ordem correta de tratamento (Limpeza -> Normalização -> Escala) antes de tocar na IA.' 
  },

  'Auto-Correction Feedback': { 
    type:'logic', 
    inputs:['prediction', 'actual_result'], 
    outputs:['bias_adjustment'], 
    logic:'def adjust(p, r): return p - r', 
    function:'Auto-Ajuste', 
    synergies:['Reinforcement Learning', 'Post-Mortem Analysis'], 
    desc:'O Corretor: Uma função que compara o tiro com o alvo. Se a IA errou para cima, ela gera um ajuste negativo automático para o próximo ciclo de predição.' 
  },

  'Inference Latency Guard': { 
    type:'logic', 
    inputs:['execution_time'], 
    outputs:['performance_alert'], 
    logic:'def monitor(t): return "SLOW" if t > 100ms else "OK"', 
    function:'Monitor de Performance', 
    synergies:['Event Loop', 'Latency Monitor'], 
    desc:'O Guarda-Tempo: Uma função que monitora quanto tempo o modelo de IA levou para "pensar". Se o Transformer demorar demais, ele sugere trocar por um modelo mais leve.' 
  },
'Tensor Pipeline Bridge': { 
    type:'logic', 
    inputs:['clean_data'], 
    outputs:['tensor_ready'], 
    logic:'Python: np.array(data).astype("float32").reshape(-1, 1)', 
    function:'Ponte de Dados -> IA', 
    synergies:['Data Cleaning', 'Neural Network', 'Transformer'], 
    desc:'O Adaptador de Cérebro: Conecta o bloco de Limpeza diretamente à Rede Neural. Ele garante que o formato do dado (shape) seja exatamente o que a IA espera para não dar erro de execução.' 
  },

  'Feature-to-Model Linker': { 
    type:'logic', 
    inputs:['selected_features'], 
    outputs:['model_input_stream'], 
    logic:'C#: var input = features.Select(f => f.Value).ToArray();', 
    function:'Vinculador de Atributos', 
    synergies:['Feature Selection', 'Linear Regression', 'XGBoost'], 
    desc:'O Organizador de Atributos: Pega apenas as melhores colunas do Feature Selection e as entrega "de bandeja" para o modelo de Regressão. Sinergia total para evitar o processamento de linguiça.' 
  },

  'Threshold Activation Function': { 
    type:'logic', 
    inputs:['prediction_output'], 
    outputs:['trigger_signal'], 
    logic:'JS: return prediction > 0.95 ? "FIRE" : "WAIT";', 
    function:'Gatilho de Decisão', 
    synergies:['Sigmoid Function', 'API Pusher (Webhook)'], 
    desc:'O Dedo no Gatilho: Conecta o resultado do Modelo ao Webhook de execução. Ele só libera o sinal de "FIRE" se a IA tiver uma certeza absoluta (acima de 95%).' 
  },

  'Cross-Model Ensemble Merger': { 
    type:'logic', 
    inputs:['pred_model_1', 'pred_model_2'], 
    outputs:['consensus_signal'], 
    logic:'Python: final = (m1 * 0.7) + (m2 * 0.3)', 
    function:'Fusor de Inteligência', 
    synergies:['Random Forest', 'Neural Network', 'Consensus Engine'], 
    desc:'O Unificador: Faz a sinergia entre dois modelos diferentes. Ele dá mais peso para o modelo que está acertando mais e gera uma decisão única muito mais forte.' 
  },

  'Data Drift Guard Function': { 
    type:'logic', 
    inputs:['live_data', 'training_stats'], 
    outputs:['is_context_valid'], 
    logic:'Python: ks_test(live, train) < 0.05', 
    function:'Segurança de Contexto', 
    synergies:['Live API Connector', 'Standard Deviation (Z-Score)'], 
    desc:'O Fiscal de Realidade: Conecta os dados ao vivo com as estatísticas de treino. Se o mercado mudar muito (Drift), ele avisa o sistema que o modelo atual pode falhar.' 
  },

  'Post-Action Feedback Loop': { 
    type:'logic', 
    inputs:['actual_result'], 
    outputs:['model_update_signal'], 
    logic:'Python: model.partial_fit(x, y)', 
    function:'Função de Aprendizado', 
    synergies:['Post-Mortem Analysis', 'Reinforcement Learning'], 
    desc:'O Professor: Pega o resultado real do mercado e envia de volta para a IA para que ela aprenda com o erro instantaneamente. É a sinergia entre o passado e o futuro.' 
  },

  'Latency-Aware Execution': { 
    type:'logic', 
    inputs:['signal_ready', 'current_latency'], 
    outputs:['optimized_webhook'], 
    logic:'JS: if(ping < 50ms) sendFullPayload(); else sendCompact();', 
    function:'Otimizador de Disparo', 
    synergies:['API Pusher (Webhook)', 'Latency Monitor'], 
    desc:'O Ajustador de Velocidade: Conecta o sinal pronto ao disparo, ajustando o tamanho do pacote de dados baseado na velocidade da internet para garantir que o tiro chegue primeiro.' 
  },
'Adaptive Confidence Scaler': { 
    type:'logic', 
    inputs:['model_prediction', 'historical_accuracy'], 
    outputs:['scaled_confidence'], 
    logic:'Python: return pred * (acc / 100)', 
    function:'Ajuste de Confiança Real', 
    synergies:['Neural Network', 'Post-Mortem Analysis', 'Quick Backtester'], 
    desc:'O Calibrador de Certeza: Uma função que ajusta a confiança da IA baseada no sucesso passado. Se o modelo diz 100%, mas o Backtester diz que ele só acerta 80%, esta função reduz o sinal para 80% para evitar excesso de confiança.' 
  },

  'Feature Weight Decay': { 
    type:'logic', 
    inputs:['raw_features', 'time_delta'], 
    outputs:['weighted_features'], 
    logic:'C#: features.Select(f => f * Math.Exp(-lambda * time))', 
    function:'Poda de Relevância Temporal', 
    synergies:['Time-Decay Weighting', 'Linear Regression'], 
    desc:'O Esquecedor Seletivo: Uma função que faz os dados antigos "perderem força". Ela conecta o histórico ao modelo de Regressão, garantindo que o sinal de 1 hora atrás não tenha o mesmo peso do sinal de 1 minuto atrás.' 
  },

  'Categorical Encoding Map': { 
    type:'logic', 
    inputs:['text_categories'], 
    outputs:['encoded_vectors'], 
    logic:'Python: pd.get_dummies(data) | LabelEncoder().fit_transform(x)', 
    function:'Tradutor de Texto para IA', 
    synergies:['Data Parsing', 'Neural Network', 'One-Hot Encoding'], 
    desc:'O Codificador de Categóricos: Transforma nomes e textos em números que a Rede Neural consegue processar. É a sinergia perfeita entre o dado bruto e o neurônio matemático.' 
  },

  'Parallel Model Sync (Vote)': { 
    type:'logic', 
    inputs:['pred_1', 'pred_2', 'pred_3'], 
    outputs:['final_consensus'], 
    logic:'JS: const vote = Math.sign(p1 + p2 + p3)', 
    function:'Sincronizador de Votação', 
    synergies:['Consensus Engine (The Judge)', 'Go/Golang Concurrency'], 
    desc:'O Juiz de Paz: Sincroniza três ou mais modelos rodando em paralelo. Ele espera o resultado de todos e só libera o sinal final se houver consenso, aumentando drasticamente a segurança do sniper.' 
  },

  'Hyper-Parameter Auto-Tuner': { 
    type:'logic', 
    inputs:['model_performance', 'param_grid'], 
    outputs:['best_params'], 
    logic:'Python: GridSearchCV(model, params).fit(X, y)', 
    function:'Otimizador de IA', 
    synergies:['Auto-Model Selector', 'Reinforcement Learning'], 
    desc:'O Mecânico de Modelos: Uma função que testa automaticamente diferentes configurações para a sua IA (ex: profundidade da árvore ou taxa de aprendizado) e escolhe a que dá mais lucro.' 
  },

  'Reward-Based Stop Loss': { 
    type:'logic', 
    inputs:['current_pnl', 'risk_limit'], 
    outputs:['kill_signal'], 
    logic:'Python: reward = 1 if pnl > 0 else -2; if(sum(rewards) < limit) stop()', 
    function:'Filtro de Recompensa/Risco', 
    synergies:['Reward Calculator', 'Bankroll Protector', 'Emergency Kill Switch'], 
    desc:'O Sensor de Perda: Conecta o lucro real à segurança do sistema. Se a sequência de "recompensas" for negativa, ele aciona o Kill Switch para proteger o que sobrou da banca.' 
  },

  'JSON-to-Tensor Formatter': { 
    type:'logic', 
    inputs:['json_data'], 
    outputs:['tensor_object'], 
    logic:'Python: torch.tensor(json.loads(data)).to(device)', 
    function:'Formatador Final', 
    synergies:['JSON Input', 'Neural Network', 'GPU Acceleration'], 
    desc:'O Empacotador de Alta Performance: Transforma o JSON diretamente em um Tensor pronto para ser processado pela placa de vídeo (GPU). Sinergia total para velocidade máxima.' 
  },
'Sigmoid Logistic Function': { 
    type:'logic', 
    inputs:['linear_score'], 
    outputs:['probability_0_1'], 
    logic:'Math: 1 / (1 + exp(-x))', 
    function:'Conversão Não-Linear', 
    synergies:['Linear Regression', 'Neural Network', 'Logistic Regression'], 
    desc:'O Conversor de Certeza: Pega o resultado de uma Regressão Linear (que pode ser qualquer número) e o "esmaga" em uma curva S entre 0 e 1. Transforma scores em porcentagem real.' 
  },

  'Bayesian Evidence Updater': { 
    type:'logic', 
    inputs:['prior_prob', 'new_evidence'], 
    outputs:['posterior_prob'], 
    logic:'Math: (P(B|A) * P(A)) / P(B)', 
    function:'Probabilidade Recursiva', 
    synergies:['Bayesian Updater', 'Live API Connector', 'Poisson Distribution'], 
    desc:'O Atualizador de Crença: Usa o Teorema de Bayes para ajustar a probabilidade conforme o jogo acontece. Se o "Prior" era 50% e um evento ocorre, ele recalcula a nova chance de vitória.' 
  },

  'Polynomial Basis Mapping': { 
    type:'logic', 
    inputs:['linear_input'], 
    outputs:['non_linear_features'], 
    logic:'Math: f(x) = [x, x², x³...]', 
    function:'Expansão de Dimensão', 
    synergies:['Polynomial Regression', 'Polynomial Features', 'Linear Regression'], 
    desc:'O Curvador de Dados: Transforma uma linha reta em uma curva complexa. Permite que a Regressão Linear entenda tendências que sobem e descem (não-lineares) no mercado.' 
  },

  'Log-Likelihood Estimator': { 
    type:'logic', 
    inputs:['prediction', 'actual_result'], 
    outputs:['fit_score'], 
    logic:'Math: Σ(y * log(p) + (1-y) * log(1-p))', 
    function:'Ajuste de Máxima Verossimilhança', 
    synergies:['Neural Network', 'Reinforcement Learning', 'Quick Backtester'], 
    desc:'O Medidor de Verdade: Calcula o quanto a sua IA "acreditou" no resultado certo. Essencial para treinar o modelo a ser mais agressivo onde ele tem mais chance de acertar.' 
  },

  'Softmax Multi-Probability': { 
    type:'logic', 
    inputs:['logits_array'], 
    outputs:['prob_distribution'], 
    logic:'Math: exp(xi) / Σ(exp(xj))', 
    function:'Normalização Não-Linear Múltipla', 
    synergies:['Neural Network', 'Transformer', 'Consensus Engine'], 
    desc:'O Distribuidor de Peso: Usado quando você tem mais de 2 opções (ex: Casa, Empate, Fora). Garante que a soma das probabilidades de todos os resultados seja sempre 100%.' 
  },

  'Kelly Criterion Optimizer': { 
    type:'logic', 
    inputs:['win_probability', 'odds'], 
    outputs:['bankroll_fraction'], 
    logic:'Math: (p * b - q) / b', 
    function:'Gestão de Banca Linear', 
    synergies:['Expected Value (EV+)', 'Bankroll Protector', 'API Pusher (Webhook)'], 
    desc:'A Fórmula da Fortuna: Calcula matematicamente quanto da sua banca você deve arriscar baseado na sua vantagem (Edge). Evita que você aposte demais em algo incerto.' 
  },

  'Standard Deviation Z-Filter': { 
    type:'logic', 
    inputs:['price_stream'], 
    outputs:['is_outlier'], 
    logic:'Math: (x - μ) / σ > threshold', 
    function:'Filtro de Anomalia Linear', 
    synergies:['Standard Deviation (Z-Score)', 'Data Cleaning', 'Outlier Sanitizer'], 
    desc:'O Detector de Zebra: Analisa se o movimento atual do preço está dentro da normalidade estatística. Se o Z-Score for alto demais, ele avisa que o sinal é perigoso ou uma anomalia.' 
  },
'C-Level Math Kernel': { 
    type:'logic', 
    inputs:['raw_vector'], 
    outputs:['math_result'], 
    logic:'C: void compute(float* v) { *v = sqrtf(*v); }', 
    function:'Cálculo de Baixo Nível', 
    synergies:['Normalization', 'Standard Deviation (Z-Score)'], 
    desc:'O Motor Bruto: Usa a linguagem C para cálculos matemáticos diretos na memória. Código: "float fast_inv_sqrt(float x)". Ideal para quando a CPU precisa processar milhões de cálculos por segundo.' 
  },

  'C++ Vectorized Dot Product': { 
    type:'logic', 
    inputs:['weights', 'inputs'], 
    outputs:['dot_product'], 
    logic:'C++: std::inner_product(w.begin(), w.end(), i.begin(), 0.0);', 
    function:'Matemática de IA em C++', 
    synergies:['Neural Network', 'Linear Weighted Sum', 'GPU Acceleration'], 
    desc:'O Neurônio de Elite: Usa C++ e instruções de hardware (SIMD) para multiplicar matrizes de IA com velocidade máxima. Sinergia total com Redes Neurais que exigem tempo de resposta zero.' 
  },

  'Java Multithreaded Executor': { 
    type:'logic', 
    inputs:['task_queue'], 
    outputs:['processed_results'], 
    logic:'Java: ExecutorService executor = Executors.newFixedThreadPool(10);', 
    function:'Processamento Paralelo Java', 
    synergies:['Auto-Model Selector', 'Parallel Processor'], 
    desc:'O Orquestrador Robusto: Usa a Máquina Virtual Java (JVM) para rodar vários modelos ao mesmo tempo em diferentes núcleos do processador. Garante que se um modelo travar, o sistema continua vivo.' 
  },

  'C++ Non-Linear Optimizer': { 
    type:'logic', 
    inputs:['loss_function', 'params'], 
    outputs:['optimized_params'], 
    logic:'C++: nlopt::optimize(opt, x, min_f);', 
    function:'Otimização Não-Linear', 
    synergies:['Polynomial Regression', 'Reinforcement Learning'], 
    desc:'O Ajustador de Curvas: Biblioteca de C++ especializada em encontrar o ponto perfeito em funções não-lineares complexas. É o que calibra a mira do sniper em mercados caóticos.' 
  },

  'Java Enterprise Data Stream': { 
    type:'data', 
    inputs:['external_source'], 
    outputs:['stream_payload'], 
    logic:'Java: Stream.of(data).filter(x -> x.isValid()).collect(Collectors.toList());', 
    function:'Fluxo de Dados Java', 
    synergies:['Live API Connector', 'Kafka Stream'], 
    desc:'O Filtro de Dados: Usa a API de Streams do Java para limpar e organizar milhões de linhas de dados vindo da internet antes de enviar para a IA. Seguro, escalável e rápido.' 
  },

  'C Memory Buffer (Pointer)': { 
    type:'logic', 
    inputs:['data_chunk'], 
    outputs:['memory_address'], 
    logic:'C: uint8_t* buffer = (uint8_t*)malloc(size); memcpy(buffer, data, size);', 
    function:'Gestão de Memória Local', 
    synergies:['WebSocket Stream', 'Redis Cache Input'], 
    desc:'O Acesso Direto: Usa ponteiros de C para manipular dados diretamente na memória RAM, pulando camadas lentas do sistema operacional. É a forma mais rápida de receber dados do WebSocket.' 
  },

  'Java Object-Oriented Model': { 
    type:'logic', 
    inputs:['model_data'], 
    outputs:['java_object'], 
    logic:'Java: public class SniperModel { private double probability; ... }', 
    function:'Estrutura de Classe Java', 
    synergies:['JSON Export', 'Model Registry'], 
    desc:'O Molde Profissional: Organiza toda a lógica da IA em objetos Java reutilizáveis. Ideal para criar sistemas grandes que precisam ser mantidos por muito tempo com segurança total.' 
  },
'Multivariate Probability Aggregator': { 
    type:'logic', 
    inputs:['prob_list', 'weights'], 
    outputs:['unified_prob'], 
    logic:'Python: 1 / (1 + exp(-sum(w * logit(p))))', 
    function:'Agregação de Certeza', 
    synergies:['Consensus Engine (The Judge)', 'Neural Network', 'Softmax'], 
    desc:'O Unificador de Opiniões: Uma função que recebe probabilidades de vários modelos (ex: RNN, Transformer e Linear) e as funde em uma única nota de 0 a 100%. Sinergia total para evitar que um modelo isolado erre o alvo.' 
  },

  'Exponential Probability Smoother': { 
    type:'logic', 
    inputs:['prob_stream', 'alpha_factor'], 
    outputs:['smoothed_prob'], 
    logic:'Math: S_t = α * x_t + (1 - α) * S_t-1', 
    function:'Suavização de Ruído', 
    synergies:['Time Series Data', 'Live API Connector', 'RNN'], 
    desc:'O Amortecedor de Oscilação: Impede que o sniper atire por causa de um "pico" momentâneo de sorte ou azar. Ele suaviza a curva de probabilidade, focando na tendência real do evento.' 
  },

  'Joint Probability Mixer (AND Logic)': { 
    type:'logic', 
    inputs:['prob_a', 'prob_b'], 
    outputs:['intersection_prob'], 
    logic:'Math: P(A ∩ B) = P(A) * P(B|A)', 
    function:'Cálculo de Eventos Cruzados', 
    synergies:['Expected Value (EV+)', 'Double Poisson'], 
    desc:'O Analista de Combinação: Calcula a chance de duas coisas acontecerem ao mesmo tempo (ex: Time A ganha E sai mais de 2 gols). Vital para snipes de alta lucratividade.' 
  },

  'Entropy Confidence Filter': { 
    type:'logic', 
    inputs:['prob_distribution'], 
    outputs:['entropy_score'], 
    logic:'Math: H(p) = -Σ p(x) log p(x)', 
    function:'Medidor de Confusão da IA', 
    synergies:['Neural Network', 'Transformer', 'Sigmoid Function'], 
    desc:'O Sensor de Incerteza: Mede o quão "confusa" a IA está. Se a entropia for alta (IA indecisa), o bloco bloqueia o disparo, mesmo que a probabilidade pareça boa. Segurança matemática pura.' 
  },

  'Markov Chain Transition Map': { 
    type:'logic', 
    inputs:['current_state', 'transition_matrix'], 
    outputs:['next_state_prob'], 
    logic:'Math: P(X_n+1 | X_n)', 
    function:'Previsão de Próximo Passo', 
    synergies:['RNN', 'LSTM', 'Time Series Data'], 
    desc:'O Mapeador de Futuro: Calcula a chance do mercado mudar de estado (ex: de "Calmo" para "Explosivo"). Cria a sinergia perfeita com modelos sequenciais para prever o próximo movimento.' 
  },

  'Non-Linear Kernel Mapper (RBF)': { 
    type:'logic', 
    inputs:['linear_features'], 
    outputs:['high_dim_prob'], 
    logic:'Math: K(x, y) = exp(-γ ||x-y||²)', 
    function:'Mapeamento em Alta Dimensão', 
    synergies:['Polynomial Regression', 'Support Vector Machine (SVM)'], 
    desc:'O Tradutor de Complexidade: Pega dados que parecem impossíveis de resolver em uma linha reta e os projeta em um espaço curvo onde a probabilidade de acerto aparece claramente.' 
  },

  'Predictive Drift Synchronizer': { 
    type:'logic', 
    inputs:['model_prob', 'market_velocity'], 
    outputs:['real_time_adjusted_prob'], 
    logic:'Math: P_adj = (P_m * V_m) / sqrt(1 + V_m²)', 
    function:'Sincronização com o Mercado', 
    synergies:['Live API Connector', 'WebSocket Stream', 'Bayesian Updater'], 
    desc:'O Ajustador de Velocidade: Sincroniza a probabilidade da IA com a velocidade real do mercado. Se o mercado acelerar, a função ajusta a sensibilidade do sniper para não perder o timing.' 
  },
'Euclidean Distance (L2)': { 
    type:'logic', 
    inputs:['vector_a', 'vector_b'], 
    outputs:['distance_score'], 
    logic:'Math: sqrt(Σ(ai - bi)²)', 
    function:'Cálculo de Proximidade', 
    synergies:['K-Nearest Neighbors', 'K-Means', 'Anomaly Detection'], 
    desc:'O Medidor de Similaridade: Calcula a distância geométrica entre dois pontos. Ideal para ver se o cenário atual é fisicamente parecido com um cenário vencedor do passado.' 
  },

  'Cosine Similarity (NLP)': { 
    type:'logic', 
    inputs:['embedding_a', 'embedding_b'], 
    outputs:['similarity_percent'], 
    logic:'Math: (A·B) / (||A|| * ||B||)', 
    function:'Comparação Semântica', 
    synergies:['BERT Embeddings', 'Text Dataset', 'Transformer'], 
    desc:'O Comparador de Significado: Mede o ângulo entre dois vetores. É a fórmula que o BERT usa para saber se duas notícias ou frases dizem a mesma coisa, mesmo com palavras diferentes.' 
  },

  'Manhattan Distance (L1)': { 
    type:'logic', 
    inputs:['vector_a', 'vector_b'], 
    outputs:['distance_score'], 
    logic:'Math: Σ|ai - bi|', 
    function:'Distância Robusta', 
    synergies:['Data Cleaning', 'Robust Scaling'], 
    desc:'O Medidor de Grade: Diferente da Euclidiama, ela é menos sensível a "Outliers" (erros absurdos). Ótima para comparar dados de mercado financeiro que oscilam muito rápido.' 
  },

  'Data Type Casting (Safe)': { 
    type:'logic', 
    inputs:['raw_input'], 
    outputs:['typed_output'], 
    logic:'C#: (double)val | Py: float(val) | JS: Number(val)', 
    function:'Conversão de Tipo', 
    synergies:['Data Parsing', 'JSON Input', 'CSV Input'], 
    desc:'O Tradutor de Formato: Garante que um texto (ex: "1.50") vire um número real (float) para que a IA não trave. Possui proteção contra valores nulos ou erros de caractere.' 
  },

  'One-Hot Encoder Converter': { 
    type:'logic', 
    inputs:['categorical_label'], 
    outputs:['numerical_vector'], 
    logic:'Math: [0, 0, 1, 0...]', 
    function:'Vetorização de Categorias', 
    synergies:['Neural Network', 'Decision Tree', 'One-Hot Encoding'], 
    desc:'O Transformador de Etiquetas: Transforma nomes (ex: "Time A", "Time B") em colunas binárias. Essencial para que a Rede Neural consiga processar nomes de times ou ativos.' 
  },

  'Min-Max Range Scaler': { 
    type:'logic', 
    inputs:['value'], 
    outputs:['scaled_value'], 
    logic:'Math: (x - min) / (max - min)', 
    function:'Conversão de Escala Linear', 
    synergies:['Normalization', 'Linear Regression', 'Sigmoid Function'], 
    desc:'O Nivelador de Campo: Coloca todos os números entre 0 e 1. Garante que uma odd de 1.50 e um volume de 1.000.000 tenham a mesma importância visual para o modelo.' 
  },

  'JSON-to-Table Pivot': { 
    type:'logic', 
    inputs:['json_stream'], 
    outputs:['tabular_dataframe'], 
    logic:'Python: pd.json_normalize(data)', 
    function:'Reestruturação de Dados', 
    synergies:['JSON Input', 'Data Parsing', 'Feature Engineering'], 
    desc:'O Organizador de Estrutura: Pega um JSON complexo e "achata" ele em uma tabela. É a sinergia final que prepara dados da Web para entrar na esteira de pré-processamento.' 
  },
'Jaccard Similarity (Sets)': { 
    type:'logic', 
    inputs:['set_a', 'set_b'], 
    outputs:['intersection_score'], 
    logic:'Math: (A ∩ B) / (A ∪ B)', 
    function:'Comparação de Conjuntos', 
    synergies:['Text Dataset', 'Data Cleaning', 'Feature Selection'], 
    desc:'O Medidor de Sobreposição: Verifica o quanto dois grupos de dados são parecidos. Excelente para comparar "listas de eventos" ou ver se dois times têm históricos de adversários em comum.' 
  },

  'Hamming Distance (Binary)': { 
    type:'logic', 
    inputs:['binary_a', 'binary_b'], 
    outputs:['bit_diff_count'], 
    logic:'Math: Σ (ai ≠ bi)', 
    function:'Comparação de Bits', 
    synergies:['One-Hot Encoding', 'Classification Models'], 
    desc:'O Comparador de Diferenças: Conta quantas posições são diferentes entre dois códigos. Muito rápido para comparar categorias de IA ou "assinaturas digitalizadas" de comportamentos de mercado.' 
  },

  'StandardScaler (Z-Score Conv)': { 
    type:'logic', 
    inputs:['raw_value'], 
    outputs:['z_standardized_value'], 
    logic:'Math: (x - μ) / σ', 
    function:'Padronização Estatística', 
    synergies:['Standard Deviation (Z-Score)', 'SVM', 'KNN'], 
    desc:'O Ajustador de Curva: Diferente da Min-Max, esta conversão foca na média e no desvio padrão. É a melhor forma de preparar dados para modelos sensíveis à dispersão, como o SVM.' 
  },

  'PCA Dimension Reducer': { 
    type:'logic', 
    inputs:['high_dim_data'], 
    outputs:['low_dim_components'], 
    logic:'Math: Eigenvector Decomposition', 
    function:'Conversão de Alta Complexidade', 
    synergies:['Feature Selection', 'Data Visualization', 'Neural Network'], 
    desc:'O Compactador Inteligente: Transforma 100 colunas de dados em apenas 3 ou 5 "Componentes Principais" sem perder a essência. Faz o seu Sniper rodar muito mais rápido com menos carga.' 
  },

  'Logarithmic Transformer': { 
    type:'logic', 
    inputs:['exponential_data'], 
    outputs:['linearized_data'], 
    logic:'Math: log(x + 1)', 
    function:'Linearização de Tendência', 
    synergies:['Linear Regression', 'Polynomial Regression'], 
    desc:'O Suavizador de Explosões: Transforma crescimentos exponenciais (pumps) em linhas retas. Ajuda a Regressão Linear a prever movimentos que antes pareciam "fora da curva".' 
  },

  'Fast Fourier Transform (FFT)': { 
    type:'logic', 
    inputs:['time_series_signal'], 
    outputs:['frequency_spectrum'], 
    logic:'Math: Σ xn * e^(-i2πkn/N)', 
    function:'Conversão Tempo-Frequência', 
    synergies:['Time Series Data', 'Audio Dataset', 'RNN'], 
    desc:'O Analista de Ciclos: Converte dados de tempo em ondas de frequência. Perfeito para detectar "ciclos" no mercado (ex: a cada 2 horas o preço oscila). É a sinergia total com Séries Temporais.' 
  },

  'Quantile Discretizer (Binning)': { 
    type:'logic', 
    inputs:['continuous_value'], 
    outputs:['categorical_bin'], 
    logic:'Math: Sort -> Group into N bins', 
    function:'Conversão Numérico-Categórica', 
    synergies:['Decision Tree', 'Random Forest', 'XGBoost'], 
    desc:'O Classificador de Faixas: Transforma números contínuos em "categorias" (ex: Baixo, Médio, Alto). Isso torna os modelos de árvore muito mais precisos e menos propensos a erros de ruído.' 
  },
 'Exponential Time-Decay (Lambda)': { 
    type:'logic', 
    inputs:['raw_value', 'time_elapsed'], 
    outputs:['decayed_value'], 
    logic:'Math: V * exp(-λ * t)', 
    function:'Depreciação de Relevância', 
    synergies:['Time-Decay Weighting', 'Polynomial Regression', 'Bayesian Updater'], 
    desc:'O Envelhecedor de Dados: Faz um dado perder valor conforme o tempo passa. SINERGIA: Garante que uma previsão feita há 5 minutos tenha menos peso que uma feita há 5 segundos.' 
  },

  'Rate-Limit Token Bucket': { 
    type:'logic', 
    inputs:['execution_request'], 
    outputs:['is_allowed', 'wait_time'], 
    logic:'Math: if(tokens > 0) { tokens--; allow(); }', 
    function:'Controle de Vazão', 
    synergies:['API Pusher (Webhook)', 'Execution Quota'], 
    desc:'O Limitador de Disparos: Controla quantos snipes podem ser dados por segundo. SINERGIA: Protege sua conta de ser banida por excesso de requisições na API (Rate Limit).' 
  },

  'Time-Window Scheduler': { 
    type:'logic', 
    inputs:['system_clock'], 
    outputs:['is_active_window'], 
    logic:'Math: if(now >= start && now <= end) return true', 
    function:'Filtro de Horário', 
    synergies:['Live API Connector', 'Main Event Loop'], 
    desc:'O Cronômetro de Operação: Define janelas exatas para o sniper trabalhar (ex: apenas nos últimos 10 minutos do jogo). Bloqueia qualquer ação fora do horário programado.' 
  },

  'Hard-Limit Clamp (Saturation)': { 
    type:'logic', 
    inputs:['raw_input'], 
    outputs:['clamped_value'], 
    logic:'Math: max(min_val, min(max_val, x))', 
    function:'Trava de Amplitude', 
    synergies:['Normalization', 'Kelly Criterion', 'Sigmoid Function'], 
    desc:'O Limitador de Escopo: Garante que um valor nunca ultrapasse um teto ou piso (ex: nunca apostar mais que 5% da banca, não importa o que a IA diga).' 
  },

  'Velocity Delta Limit': { 
    type:'logic', 
    inputs:['value_t1', 'value_t0', 'delta_t'], 
    outputs:['acceleration_score'], 
    logic:'Math: (v1 - v0) / dt', 
    function:'Monitor de Aceleração', 
    synergies:['Standard Deviation (Z-Score)', 'WebSocket Stream'], 
    desc:'O Radar de Velocidade: Mede o quão rápido o preço ou odd está mudando. Se a aceleração for alta demais (pânico no mercado), o limite pode travar o sniper por segurança.' 
  },

  'Timeout Deadlock Guard': { 
    type:'logic', 
    inputs:['last_data_timestamp'], 
    outputs:['is_stale_alert'], 
    logic:'Math: if(now - last > timeout) restart()', 
    function:'Vigilante de Conexão', 
    synergies:['Heartbeat Monitor', 'WebSocket Stream', 'Live API Connector'], 
    desc:'O Guarda de Inatividade: Se os dados pararem de chegar por X segundos, ele entende que a conexão "congelou" e reinicia o fluxo para o sniper não trabalhar com dados velhos.' 
  },

  'Drawdown Mathematical Cap': { 
    type:'logic', 
    inputs:['current_equity', 'peak_equity'], 
    outputs:['drawdown_percent'], 
    logic:'Math: (peak - current) / peak', 
    function:'Limite de Queda', 
    synergies:['Bankroll Protector', 'Emergency Kill Switch', 'Cumulative Loss Guard'], 
    desc:'O Sensor de Queda Máxima: Calcula em tempo real o quanto você está perdendo em relação ao seu topo de capital. Se atingir o limite (ex: 10%), o sistema desliga tudo.' 
  },
'Quantum Superposition State': { 
    type:'logic', 
    inputs:['model_list', 'input_data'], 
    outputs:['multi_state_probabilities'], 
    logic:'Math: Ψ = Σ Ci * |φi>', 
    function:'Processamento Multi-Cenário', 
    synergies:['Auto-Model Selector', 'Monte Carlo Simulation', 'Parallel Processor'], 
    desc:'O Estado de Sobreposição: Em vez de escolher um modelo, ele roda todos simultaneamente em "estados sobrepostos". Ele mantém todas as possibilidades de lucro vivas até o momento da observação (disparo).' 
  },

  'Quantum Entanglement (Correlation)': { 
    type:'logic', 
    inputs:['asset_a', 'asset_b'], 
    outputs:['entanglement_score'], 
    logic:'Math: Bell_State |00> + |11>', 
    function:'Correção por Entrelaçamento', 
    synergies:['Pearson Correlation', 'Cross-Market Arbitrage Link'], 
    desc:'O Entrelaçamento de Dados: Detecta quando dois mercados estão conectados de forma invisível. Se o "Ativo A" sobe, o "Ativo B" reage instantaneamente, não importa a distância. Identifica correlações fantasmagóricas.' 
  },

  'Nuclear Signal Fusion': { 
    type:'logic', 
    inputs:['weak_signals_list'], 
    outputs:['critical_mass_signal'], 
    logic:'Math: E = mc² (Signal_Energy)', 
    function:'Fusão de Sinais', 
    synergies:['Ensemble Weight Blender', 'Consensus Engine (The Judge)'], 
    desc:'O Reator de Fusão: Pega 55 pequenos sinais fracos e os funde em um único sinal de "Massa Crítica" ultra-poderoso. É o que transforma pequenos indícios em um tiro de 100 pontos de confiança.' 
  },

  'Heisenberg Uncertainty Filter': { 
    type:'logic', 
    inputs:['precision_data', 'velocity_data'], 
    outputs:['optimal_balance_point'], 
    logic:'Math: Δx * Δp ≥ ℏ/2', 
    function:'Equilíbrio de Incerteza', 
    synergies:['Latency Monitor', 'Confidence Score Validator'], 
    desc:'O Filtro de Incerteza: Lida com o dilema: "Quanto mais rápido eu tento ser, menos preciso eu sou". Ele encontra o ponto exato de equilíbrio onde a velocidade do sniper não sacrifica a precisão do alvo.' 
  },

  'Radioactive Information Decay': { 
    type:'logic', 
    inputs:['news_data'], 
    outputs:['half_life_value'], 
    logic:'Math: N(t) = N0 * e^(-λt)', 
    function:'Meia-vida da Informação', 
    synergies:['Time-Decay Weighting', 'Sentiment Database'], 
    desc:'A Meia-vida do Dado: Trata a notícia como um elemento radioativo. Ele calcula o tempo exato em que uma informação "esfria" e deixa de ser perigosa ou útil para o sistema.' 
  },

  'Quantum Tunneling (Optimizer)': { 
    type:'logic', 
    inputs:['optimization_barrier'], 
    outputs:['bypass_solution'], 
    logic:'Math: T ≈ e^(-2G)', 
    function:'Salto de Barreira Lógica', 
    synergies:['Genetic Hyper-Optimizer', 'Reinforcement Learning'], 
    desc:'O Tunelamento Quântico: Permite que o algoritmo "atravesse" paredes de erros ou mínimos locais que travariam uma IA normal. Ele encontra soluções impossíveis saltando buracos na lógica.' 
  },

  'Nuclear Fission (Data Splitter)': { 
    type:'logic', 
    inputs:['complex_json_payload'], 
    outputs:['atomic_data_stream'], 
    logic:'Math: Heavy_Nucleus -> Light_Nuclei + Energy', 
    function:'Decomposição Atômica', 
    synergies:['JSON Input', 'Data Parsing', 'Feature-to-Model Linker'], 
    desc:'A Fissão de Dados: Quebra um arquivo JSON pesado e complexo em suas partículas mais fundamentais (átomos de dados). Libera energia de processamento ao simplificar o que era denso.' 
  },
'Linear Vector Extrapolation': { 
    type:'logic', 
    inputs:['current_velocity', 'acceleration'], 
    outputs:['future_point_projection'], 
    logic:'Math: P(t+1) = P(t) + V*Δt + 0.5*a*Δt²', 
    function:'Projeção Linear de Futuro', 
    synergies:['Linear Regression', 'WebSocket Stream'], 
    desc:'O Extrapolador de Tendência: Calcula onde o preço ou odd estará no próximo segundo baseado na velocidade atual. É o tiro "na frente do alvo" que snipers reais usam.' 
  },

  'Chaotic Attractor Map (Lorenz)': { 
    type:'logic', 
    inputs:['time_series_flow'], 
    outputs:['non_linear_stability_zone'], 
    logic:'Math: dx/dt = σ(y-x) | dy/dt = x(ρ-z)-y', 
    function:'Previsão de Caos Não-Linear', 
    synergies:['Polynomial Regression', 'RNN', 'Standard Deviation (Z-Score)'], 
    desc:'O Mapeador de Caos: Identifica padrões em mercados "doidos". Ele detecta se o caos atual vai convergir para um ponto de lucro ou se vai explodir em perda. Captura o futuro não-linear.' 
  },

  'Borns Rule Probability (Waveform)': { 
    type:'logic', 
    inputs:['complex_amplitude'], 
    outputs:['collapsed_probability'], 
    logic:'Math: P = |ψ|²', 
    function:'Probabilidade Quântica Real', 
    synergies:['Quantum Superposition State', 'Neural Network'], 
    desc:'O Colapsador de Realidade: Pega todas as possibilidades (onda) e as transforma em uma probabilidade real (partícula) de 0 a 100%. É a fórmula que decide qual futuro vai acontecer.' 
  },

  'Fractal Temporal Projection': { 
    type:'logic', 
    inputs:['historical_patterns'], 
    outputs:['future_self_similarity'], 
    logic:'Math: D = lim (ε->0) [log N(ε) / log(1/ε)]', 
    function:'Previsão Não-Linear Fractal', 
    synergies:['Time Series Data', 'Transformer', 'FFT'], 
    desc:'O Projetor de Ciclos Infinitos: Identifica que o padrão de 1 minuto se repete no padrão de 1 hora. Ele projeta o futuro baseando-se na geometria repetitiva do mercado.' 
  },

  'Quantum Interference Logic': { 
    type:'logic', 
    inputs:['signal_a', 'signal_b'], 
    outputs:['interfered_signal'], 
    logic:'Math: I = I1 + I2 + 2√(I1*I2)cos(φ)', 
    function:'Sinergia de Ondas Probabilísticas', 
    synergies:['Consensus Engine (The Judge)', 'Ensemble Weight Blender'], 
    desc:'O Cancelador de Ruído: Usa interferência destrutiva para apagar sinais falsos e interferência construtiva para amplificar sinais de 100 pontos. Faz os sinais fracos sumirem.' 
  },

  'Hilbert Space Mapping': { 
    type:'logic', 
    inputs:['multi_dimensional_data'], 
    outputs:['quantum_vector_state'], 
    logic:'Math: <φ|ψ> (Inner Product)', 
    function:'Probabilidade Quântica Multivariada', 
    synergies:['BERT Embeddings', 'PCA Dimension Reducer'], 
    desc:'O Mapa Infinito: Projeta todos os seus 275 blocos em um espaço matemático infinito para encontrar a correlação perfeita que uma lógica linear jamais veria.' 
  },

  'Non-Linear Momentum Gradient': { 
    type:'logic', 
    inputs:['error_surface', 'learning_rate'], 
    outputs:['future_optimization_path'], 
    logic:'Math: Δw = -η * ∇E + α * Δw(t-1)', 
    function:'Otimização de Trajetória Futura', 
    synergies:['Reinforcement Learning', 'Backpropagation Optimizer'], 
    desc:'O Corretor de Rumo: Não apenas corrige o erro, mas prevê onde o erro estará no futuro e já ajusta a mira da IA antecipadamente.' 
  },
'Velocity Estimator (v)': { 
    type:'logic', inputs:['p1', 'p0', 'dt'], outputs:['velocity'], 
    logic:'(p1 - p0) / dt', function:'Cálculo Cinemático', 
    synergies:['Linear Vector Extrapolation'], desc:'Calcula a velocidade vetorial do mercado para alimentar a extrapolação linear.' 
  },

  'Acceleration Estimator (a)': { 
    type:'logic', inputs:['v1', 'v0', 'dt'], outputs:['acceleration'], 
    logic:'(v1 - v0) / dt', function:'Cálculo Cinemático', 
    synergies:['Linear Vector Extrapolation'], desc:'Mede a aceleração (mudança de força) para prever o ponto futuro exato.' 
  },

  'Lyapunov Exponent Calc': { 
    type:'logic', inputs:['time_series'], outputs:['chaos_score'], 
    logic:'λ = lim (t->∞) 1/t log|δx(t)/δx(0)|', function:'Análise de Caos', 
    synergies:['Chaotic Attractor Map (Lorenz)'], desc:'Mede o grau de caos. Essencial para saber se o mapa de Lorenz é confiável no momento.' 
  },

  'Wavefunction Normalizer': { 
    type:'logic', inputs:['complex_amplitudes'], outputs:['normalized_psi'], 
    logic:'ψ / sqrt(Σ|ψ|²)', function:'Normalização Quântica', 
    synergies:['Borns Rule Probability (Waveform)'], desc:'Garante que a soma das probabilidades quânticas seja exatamente 1 (100%).' 
  },

  'Fractal Dimension Estimator': { 
    type:'logic', inputs:['price_data'], outputs:['fractal_d'], 
    logic:'log(N) / log(1/ε)', function:'Geometria Fractal', 
    synergies:['Fractal Temporal Projection'], desc:'Calcula a "aspereza" do mercado para prever a repetição de padrões em escalas menores.' 
  },

  'Signal Phase Aligner': { 
    type:'logic', inputs:['signal_a', 'signal_b'], outputs:['aligned_signals'], 
    logic:'Shift(S2) to match Argmax(Cross-Correlation)', function:'Sincronia de Ondas', 
    synergies:['Quantum Interference Logic'], desc:'Alinha as fases de dois sinais para que a interferência construtiva seja máxima (100 pontos).' 
  },

  'Vector Orthogonalizer (Gram-Schmidt)': { 
    type:'logic', inputs:['vector_set'], outputs:['orthogonal_basis'], 
    logic:'v - proj(u, v)', function:'Álgebra Linear Pro', 
    synergies:['Hilbert Space Mapping'], desc:'Limpa a redundância entre os 275 blocos, deixando apenas dados que não se repetem.' 
  },

  'Momentum Buffer (Gradient)': { 
    type:'logic', inputs:['current_grad', 'prev_momentum'], outputs:['new_momentum'], 
    logic:'m = β*m + (1-β)*grad', function:'Filtro de Inércia', 
    synergies:['Non-Linear Momentum Gradient'], desc:'Acumula a força do erro passado para acelerar o ajuste do sniper no futuro.' 
  },

  'State Observer (Schrödinger)': { 
    type:'logic', inputs:['market_events'], outputs:['quantum_state'], 
    logic:'Hψ = Eψ', function:'Monitoramento de Estado', 
    synergies:['Borns Rule Probability (Waveform)'], desc:'Monitora os eventos do WebSocket para definir em qual "estado" o mercado colapsou.' 
  },

  'Strange Attractor Detector': { 
    type:'logic', inputs:['lorenz_output'], outputs:['stability_alert'], 
    logic:'Check if inside Attractor A or B', function:'Topologia de Caos', 
    synergies:['Chaotic Attractor Map (Lorenz)'], desc:'Identifica se o mercado entrou em um "atrator" (ciclo vicioso de perda ou ganho).' 
  },

  'Self-Similarity Scorer': { 
    type:'logic', inputs:['current_pattern', 'historical_fractal'], outputs:['match_score'], 
    logic:'R(h, τ) / R(h, 0)', function:'Estatística Fractal', 
    synergies:['Fractal Temporal Projection'], desc:'Diz o quanto o gráfico de agora é uma cópia perfeita de um gráfico lucrativo do passado.' 
  },

  'Constructive Gain Filter': { 
    type:'logic', inputs:['interference_pattern'], outputs:['amplified_signal'], 
    logic:'if(cos(φ) > 0) gain++', function:'Amplificação de Sinal', 
    synergies:['Quantum Interference Logic'], desc:'Filtra apenas a parte "boa" da interferência quântica para dar o tiro de sniper.' 
  },

  'Inner Product Engine': { 
    type:'logic', inputs:['vector_psi', 'vector_phi'], outputs:['inner_product'], 
    logic:'Σ(ai * bi*)', function:'Métrica de Hilbert', 
    synergies:['Hilbert Space Mapping'], desc:'Calcula a projeção de um dado sobre o outro no espaço infinito para achar a correlação oculta.' 
  },

  'Learning Rate Decay (Predictive)': { 
    type:'logic', inputs:['epoch', 'loss_trend'], outputs:['scheduled_lr'], 
    logic:'lr / (1 + decay * epoch)', function:'Otimização Temporal', 
    synergies:['Non-Linear Momentum Gradient'], desc:'Reduz a velocidade de aprendizado automaticamente conforme o alvo é atingido.' 
  },

  'Trajectory Smoothing': { 
    type:'logic', inputs:['extrapolated_points'], outputs:['smooth_path'], 
    logic:'Savitzky-Golay Filter', function:'Refinamento de Rumo', 
    synergies:['Linear Vector Extrapolation'], desc:'Remove o ruído da trajetória prevista para o tiro não oscilar.' 
  },

  'Entropy Entropy (Chaos Guard)': { 
    type:'logic', inputs:['probability_dist'], outputs:['uncertainty'], 
    logic:'-Σ p log p', function:'Filtro de Incerteza', 
    synergies:['Chaotic Attractor Map (Lorenz)'], desc:'Avisa se o caos está tão alto que nem o mapa de Lorenz consegue prever.' 
  },

  'Amplitude Modulator': { 
    type:'logic', inputs:['psi_state', 'boost_factor'], outputs:['boosted_psi'], 
    logic:'ψ * exp(iθ)', function:'Modulação Quântica', 
    synergies:['Borns Rule Probability (Waveform)'], desc:'Aumenta a "intensidade" de um sinal de lucro antes de colapsar a probabilidade.' 
  },

  'Hurst Exponent (Long Memory)': { 
    type:'logic', inputs:['time_series'], outputs:['h_index'], 
    logic:'E[R(n)/S(n)] = C * n^H', function:'Memória de Longo Prazo', 
    synergies:['Fractal Temporal Projection'], desc:'Mede se o mercado tem memória. H > 0.5 indica que a tendência vai continuar.' 
  },

  'Decoherence Detector': { 
    type:'logic', inputs:['quantum_logic_gate'], outputs:['noise_level'], 
    logic:'1 - Fidelity(ρ)', function:'Segurança Quântica', 
    synergies:['Quantum Interference Logic'], desc:'Mede o quanto o "ruído" externo está estragando a sua lógica de interferência.' 
  },

  'Tensor Projection (Hilbert)': { 
    type:'logic', inputs:['multi_block_data'], outputs:['tensor_state'], 
    logic:'A ⊗ B (Kronecker Product)', function:'Expansão Espacial', 
    synergies:['Hilbert Space Mapping'], desc:'Une múltiplos blocos em uma única super-matriz para análise quântica.' 
  },

  'Gradient Clipping (Safety)': { 
    type:'logic', inputs:['raw_gradient'], outputs:['clipped_gradient'], 
    logic:'min(max_val, grad)', function:'Segurança de Ajuste', 
    synergies:['Non-Linear Momentum Gradient'], desc:'Impede que o erro futuro seja tão grande que "quebre" a lógica da IA.' 
  },

  'Target Lead Predictor': { 
    type:'logic', inputs:['future_point', 'latency'], outputs:['lead_adjustment'], 
    logic:'P_future + (V * latency)', function:'Compensação de Atraso', 
    synergies:['Linear Vector Extrapolation'], desc:'Ajusta a mira do sniper para compensar o lag da sua internet.' 
  },

  'Bifurcation Point Finder': { 
    type:'logic', inputs:['chaos_flow'], outputs:['bifurcation_event'], 
    logic:'δx/δμ = 0', function:'Detecção de Mudança', 
    synergies:['Chaotic Attractor Map (Lorenz)'], desc:'Detecta o milissegundo exato onde o mercado vai escolher entre subir ou cair drasticamente.' 
  },

  'Probability Density Estimator': { 
    type:'logic', inputs:['psi_samples'], outputs:['pdf_curve'], 
    logic:'Kernel Density Estimation', function:'Mapeamento de Alvo', 
    synergies:['Borns Rule Probability (Waveform)'], desc:'Cria o mapa visual de onde o lucro tem maior densidade estatística.' 
  },

  'Interference Pattern Analyzer': { 
    type:'logic', inputs:['combined_signals'], outputs:['purity_score'], 
    logic:'Fourier(S1 * S2)', function:'Análise de Sincronia', 
    synergies:['Quantum Interference Logic'], desc:'Diz se a sinergia entre os modelos é real ou apenas uma coincidência de dados.' 
  },
 'Negative Reward Penalty': { 
    type:'logic', 
    inputs:['actual_loss', 'model_prediction'], 
    outputs:['weight_reduction_signal'], 
    logic:'Python: weights -= alpha * (prediction - loss)', 
    function:'Reforço Negativo', 
    synergies:['Reinforcement Learning', 'Backpropagation Optimizer'], 
    desc:'O Castigo da IA: Se o modelo der um tiro e errar, este bloco "pune" o modelo, reduzindo a importância dele no sistema para que ele não erre de novo.' 
  },

  'Drawdown Circuit Breaker': { 
    type:'logic', 
    inputs:['current_loss_percent'], 
    outputs:['hard_stop_signal'], 
    logic:'if(loss > limit) { lock_api = true; }', 
    function:'Bloqueio de Pânico', 
    synergies:['Emergency Kill Switch', 'Bankroll Protector'], 
    desc:'O Disjuntor de Emergência: Uma ação negativa imediata. Se a perda atingir um nível crítico, ele desliga o sistema e impede qualquer nova aposta por 24h.' 
  },

  'Asset Blacklist Guard': { 
    type:'logic', 
    inputs:['performance_history'], 
    outputs:['blocked_asset_list'], 
    logic:'if(fail_rate > 70%) blacklist.add(id);', 
    function:'Banimento de Ativo', 
    synergies:['Post-Mortem Analysis', 'Data Cleaning'], 
    desc:'A Lista Negra: Identifica ativos, times ou mercados que "mentem" para o modelo e os bane permanentemente da execução.' 
  },

  'Greed/Fear Index (VIX)': { 
    type:'logic', 
    inputs:['market_volatility', 'volume_delta'], 
    outputs:['market_mood_score'], 
    logic:'Math: (Volatility * 0.7) + (Volume_Acc * 0.3)', 
    function:'Índice de Sentimento', 
    synergies:['Standard Deviation (Z-Score)', 'Live API Connector'], 
    desc:'O Termômetro de Emoção: Mede se o mercado está em pânico (Medo) ou eufórico (Ganância). Ajuda a decidir se o snipe deve ser cauteloso ou agressivo.' 
  },

  'Sentiment Collision Detector': { 
    type:'logic', 
    inputs:['price_trend', 'news_sentiment'], 
    outputs:['trap_alert'], 
    logic:'if(price == UP && news == DOWN) return TRAP;', 
    function:'Detecção de Armadilha', 
    synergies:['BERT Embeddings', 'LLM Sentiment Analysis', 'Linear Regression'], 
    desc:'O Detector de Mentiras: Se o preço está subindo, mas as notícias e o sentimento são negativos, ele identifica uma "Armadilha de Touro" e bloqueia a entrada.' 
  },

  'Social Media FOMO Filter': { 
    type:'logic', 
    inputs:['twitter_volume', 'hype_keywords'], 
    outputs:['fomo_risk_level'], 
    logic:'Math: log(mention_count) / time_delta', 
    function:'Filtro de Euforia', 
    synergies:['Twitter Sentiment', 'NLP'], 
    desc:'O Filtro Anti-Manada: Detecta quando um ativo está sendo muito comentado por robôs ou pessoas eufóricas (FOMO). Ele gera uma ação negativa (bloqueio) para não entrar no topo.' 
  },

  'Model Depression Indicator': { 
    type:'logic', 
    inputs:['accuracy_slope'], 
    outputs:['performance_depression'], 
    logic:'if(slope < -0.1) mode = CONSERVATIVE;', 
    function:'Ajuste de Humor do Sistema', 
    synergies:['Auto-Model Selector', 'Quick Backtester'], 
    desc:'A Depressão do Sistema: Se a taxa de acerto do sniper cair sistematicamente, o sistema entra em modo "Depressivo" (Conservador), reduzindo o valor das apostas sozinho.' 
  },
'Fractal Input Multiplier (1x10)': { 
    type:'data', 
    inputs:['single_data_stream'], 
    outputs:['ten_data_variants'], 
    logic:'Math: Fractal_Expansion(x, n=10)', 
    function:'Multiplicação de Entrada', 
    synergies:['JSON Input', 'Data Cleaning', 'Auto-Model Selector', 'JSON Export'], 
    desc:'O Multiplicador Inicial (1x10): Pega 1 entrada de dado e a quebra em 10 dimensões diferentes. SINERGIA: Garante que a entrada de dados bruta seja tratada em 10 perspectivas antes de chegar ao modelo.' 
  },

  'Hyper-Parallel Feature Scaler (1x100)': { 
    type:'preprocessing', 
    inputs:['ten_data_variants'], 
    outputs:['hundred_feature_vectors'], 
    logic:'Math: Linear_Expansion_Matrix(10x10)', 
    function:'Escalabilidade de Tratamento', 
    synergies:['Feature Engineering', 'Normalization', 'Neural Network', 'CSV Export'], 
    desc:'A Explosão de Atributos (1x100): Transforma as 10 variantes em 100 atributos técnicos. SINERGIA: O tratamento de dados processa cada variante com pesos diferentes para alimentar a Rede Neural com precisão total.' 
  },

  'Swarm Intelligence Model (1x1k)': { 
    type:'model', 
    inputs:['hundred_feature_vectors'], 
    outputs:['thousand_predictions'], 
    logic:'Math: Monte_Carlo_Ensemble(100 -> 1000)', 
    function:'Evolução de Modelagem', 
    synergies:['XGBoost', 'Polynomial Regression', 'API REST', 'Model Registry'], 
    desc:'O Enxame de Modelos (1x1k): Faz 100 vetores gerarem 1.000 predições simultâneas. SINERGIA: O modelo principal coordena mil sub-modelos para encontrar a predição de ouro, exportando para o registro de modelos.' 
  },

  'Omni-Channel Output Nexus (1x10k)': { 
    type:'output', 
    inputs:['thousand_predictions'], 
    outputs:['ten_thousand_actions'], 
    logic:'Math: Logic_Branching(1000 -> 10000)', 
    function:'Escalabilidade de Saída', 
    synergies:['API Pusher (Webhook)', 'Data Parsing', 'Transformer', 'Database Save'], 
    desc:'O Nexo de Execução (1x10k): Transforma as 1.000 predições em 10.000 ordens de micro-ajuste ou ações. SINERGIA: A saída de dados é tão vasta que cobre todas as possibilidades de mercado de uma vez só.' 
  },

  'Self-Healing Code Optimizer': { 
    type:'logic', 
    inputs:['system_errors', 'source_code'], 
    outputs:['patched_logic'], 
    logic:'Python: AI_Code_Refactor(error_log)', 
    function:'Ajuste Processual e Erros', 
    synergies:['Try-Catch (Error Handler)', 'Data Cleaning', 'Neural Network', 'API Pusher (Webhook)'], 
    desc:'O Auto-Corretor: Analisa erros de código e se auto-corrige. SINERGIA: Ele trata os dados de erro, ajusta o modelo lógico e garante que a saída nunca falhe por bug de programação.' 
  },

  'Emotional Bio-Sync Interface': { 
    type:'logic', 
    inputs:['user_heart_rate', 'mood_input'], 
    outputs:['adaptive_ui_vibe'], 
    logic:'Math: Sentiment_Resonance(user, system)', 
    function:'Ajuste Visual e Emocional', 
    synergies:['Live API Connector', 'Normalization', 'Transformer', 'Visualization'], 
    desc:'O Sincronizador de Humor: Ajusta a cor e o ritmo do sistema baseado no seu sentimento. SINERGIA: Se você está tenso, o modelo de saída desacelera e a visualização acalma o ambiente automaticamente.' 
  },

  'Behemoth Evolutionary Bridge': { 
    type:'logic', 
    inputs:['input', 'treatment', 'model', 'output'], 
    outputs:['evolved_system_state'], 
    logic:'Math: Recursive_Evolution(Σ All)', 
    function:'Evolução Total do Sistema', 
    synergies:['JSON Input', 'Data Cleaning', 'Neural Network', 'JSON Export'], 
    desc:'A Ponte da Singularidade: Este bloco une os 4 pilares (Entrada, Tratamento, Modelo, Saída). Ele faz o sistema aprender com o todo, evoluindo a arquitetura para que 1 bit de hoje vire 10.000 bits de lucro amanhã.' 
  },
'Evolutionary Neural Nucleus': { 
    type:'model', 
    inputs:['data_stream'], outputs:['evolved_weights'], 
    logic:'Math: Σ(Inputs) * 10^n', function:'Evolução de Inteligência', 
    synergies:['JSON Input', 'Data Cleaning', 'Normalization', 'Neural Network', 'Transformer', 'RNN', 'API Pusher (Webhook)', 'JSON Export', 'Model Registry', 'Backtesting Engine'], 
    desc:'O Núcleo de Evolução (1x10): O coração do sistema. Ele conecta 10 áreas vitais para garantir que a inteligência aprenda com a entrada e ajuste a saída automaticamente.' 
  },

  'Emotional Bio-Feedback Hub': { 
    type:'logic', 
    inputs:['user_state'], outputs:['system_vibe'], 
    logic:'Math: Resonance_Sync(User, AI)', function:'Ajuste Emocional e Visual', 
    synergies:['Visualization', 'Dashboard', 'Sentiment Analysis', 'BERT Embeddings', 'Time-Decay Weighting', 'Bankroll Protector', 'Emergency Kill Switch', 'User UI Input', 'Sound Alert', 'PDF Report'], 
    desc:'O Centro de Comando Humano (1x10): Ajusta a parte visual e emocional do sistema. Ele sincroniza o seu humor com 10 blocos de segurança e visualização.' 
  },

  'Self-Healing Logic Refactor': { 
    type:'logic', 
    inputs:['error_logs'], outputs:['fixed_code'], 
    logic:'AI: Code_Autofix(Errors)', function:'Ajuste Processual e Erros', 
    synergies:['Try-Catch (Error Handler)', 'Heartbeat Monitor', 'Latency Monitor', 'Data Schema Validator', 'Parallel Processor', 'GPU Acceleration', 'Feature Selection', 'PCA', 'Database Save', 'Slack Notification'], 
    desc:'O Reparador Automático (1x10): Monitora erros de código e performance. Se algo falha, ele usa 10 conexões de infraestrutura para consertar o processo em tempo real.' 
  },

  'Hyper-Dimensional Scaler': { 
    type:'preprocessing', 
    inputs:['raw_data'], outputs:['high_dim_tensor'], 
    logic:'Math: Linear_Expansion(1x10x100)', function:'Tratamento Exponencial', 
    synergies:['Polynomial Features', 'Interaction Features', 'One-Hot Encoding', 'Label Encoding', 'Standard Deviation (Z-Score)', 'Robust Scaling', 'Min-Max Range Scaler', 'Log Transformation', 'Binning', 'Feature Engineering'], 
    desc:'O Expansor de Dados (1x10): Transforma 1 dado em uma matriz de 10 dimensões. Ele usa 10 técnicas de tratamento para garantir que a IA veja o "Caminho de Ouro".' 
  },

  'Consensus Strategy Overlord': { 
    type:'logic', 
    inputs:['multi_model_signals'], outputs:['final_snipe'], 
    logic:'Math: Weighted_Consensus(55_Models)', function:'Sinergia de Modelagem', 
    synergies:['XGBoost', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'Linear Regression', 'Support Vector Machine', 'K-Nearest Neighbors', 'Naive Bayes', 'Gradient Boosting', 'CatBoost'], 
    desc:'O Comandante de Modelos (1x10): Este bloco gerencia a sinergia entre os 10 modelos mais poderosos. Ele faz o 1 sinal virar 100 de confiança através do consenso.' 
  },

  'Global Connectivity Nexus': { 
    type:'data', 
    inputs:['external_world'], outputs:['omni_data_payload'], 
    logic:'Math: Data_Fusion(Multi_Source)', function:'Entrada e Saída Global', 
    synergies:['API REST', 'GraphQL API', 'Database SQL', 'MongoDB', 'Kafka Stream', 'BigQuery', 'AWS S3', 'Google Cloud Storage', 'Azure Blob', 'Web Scraping'], 
    desc:'O Nexo de Conectividade (1x10): Conecta o Sniper a 10 fontes de dados diferentes ao mesmo tempo. É a base para a escala de 1 para 1.000.' 
  },

  'Singularity Risk Bridge': { 
    type:'logic', 
    inputs:['all_system_data'], outputs:['safety_verdict'], 
    logic:'Math: Risk_Singularity_Check', function:'Segurança de Escala', 
    synergies:['Risk Analysis', 'Monte Carlo Simulation', 'Kelly Criterion', 'Expected Value (EV+)', 'Poisson Distribution', 'Elo Rating System', 'Bayesian Updater', 'Cross-Validation', 'Quick Backtester', 'Real-time PnL Tracker'], 
    desc:'A Ponte da Singularidade (1x10): O bloco final que valida o risco. Ele cruza 10 métricas de proteção para garantir que o 1x1000 de lucro não vire 1x1000 de perda.' 
  },
'Hyper-Synergy Compressor (1x100)': { 
    type:'logic', 
    inputs:['500_block_signals'], outputs:['100_optimized_synergies'], 
    logic:'Math: S = Σ(B_i * W_i) / Entropy', function:'Compressão de Lógica', 
    synergies:['Auto-Model Selector', 'PCA', 'Feature Selection', 'Feature Engineering', 'Normalization', 'Data Cleaning', 'Try-Catch (Error Handler)', 'Heartbeat Monitor', 'Parallel Processor', 'GPU Acceleration'], 
    desc:'O Compressor de Sinergia (1x100): Pega os sinais dos seus 500 blocos e os compacta em 100 conexões essenciais. Ele elimina o "ruído" das 5.000 sinergias para focar apenas no que dá lucro.' 
  },

  'Neural Matrix Scaler (1x1k)': { 
    type:'model', 
    inputs:['100_optimized_synergies'], outputs:['1000_neural_paths'], 
    logic:'Math: Matrix_Multiplication(100x10)', function:'Escalabilidade Neural', 
    synergies:['Neural Network', 'Transformer', 'RNN', 'LSTM', 'GRU', 'BERT Embeddings', 'CNN', 'Autoencoder', 'GAN', 'Reinforcement Learning'], 
    desc:'O Escalonador de Matriz (1x1k): Transforma as 100 sinergias comprimidas em 1.000 caminhos de decisão neural. Ele permite que o modelo processe a complexidade de 5.000 conexões sem perder velocidade.' 
  },

  'Visual Emotional Dashboard (1x10)': { 
    type:'output', 
    inputs:['1000_neural_paths'], outputs:['10_visual_metrics'], 
    logic:'Math: Emotional_Resonance_Mapping', function:'Ajuste Visual e Emocional', 
    synergies:['Visualization', 'Dashboard', 'Sentiment Analysis', 'Sound Alert', 'PDF Report', 'Telegram Bot', 'Slack Notification', 'Email Alert', 'Metrics Logger', 'User UI Input'], 
    desc:'O Dashboard Emocional (1x10): Resume os 1.000 caminhos neurais em 10 métricas visuais simples. Ele traduz a "tensão" do sistema para cores e sons que você entende de imediato.' 
  },

  'Self-Healing Code Architect (1x10)': { 
    type:'logic', 
    inputs:['system_vulnerabilities'], outputs:['optimized_codebase'], 
    logic:'AI: Genetic_Refactor(5k_Connections)', function:'Ajuste Processual e Erros', 
    synergies:['Try-Catch (Error Handler)', 'Emergency Kill Switch', 'Cumulative Loss Guard', 'Heartbeat Monitor', 'Latency Monitor', 'Data Schema Validator', 'Global Dependency Lock', 'Circuit Breaker (Anti-Panic)', 'Master Switch', 'System Mode Controller'], 
    desc:'O Arquiteto de Auto-Cura (1x10): Monitora as 5.000 conexões. Se uma sinergia falhar ou causar erro de código, ele isola o erro e reconecta o fluxo por um caminho seguro.' 
  },

  'Omni-Input Synthesizer (1x10)': { 
    type:'data', 
    inputs:['global_data_sources'], outputs:['unified_data_tensor'], 
    logic:'Math: Multi-Source_Fusion', function:'Entrada e Tratamento de Dados', 
    synergies:['JSON Input', 'CSV Input', 'XML Input', 'API REST', 'GraphQL API', 'Database SQL', 'MongoDB', 'Kafka Stream', 'Web Scraping', 'AWS S3'], 
    desc:'O Sintetizador de Entrada (1x10): Une as 10 principais fontes de dados em um único fluxo. Ele garante que a entrada de dados suporte a escala de 1 para 1.000 de processamento.' 
  },

  'Action Execution Nexus (1x10k)': { 
    type:'output', 
    inputs:['1000_neural_paths'], outputs:['10000_micro_actions'], 
    logic:'Math: Execution_Expansion_Logic', function:'Saída de Dados e Execução', 
    synergies:['API Pusher (Webhook)', 'Model Registry', 'Real-time Stream', 'Cloud Storage', 'Database Save', 'JSON Export', 'CSV Export', 'Excel Export', 'Parquet Export', 'API Endpoint'], 
    desc:'O Nexo de Execução (1x10k): O estágio final onde 1.000 caminhos neurais geram 10.000 micro-ações. É o "Snipe" de alta frequência, espalhando ordens para garantir o melhor preço.' 
  },

  'Recursive Synergy Bridge': { 
    type:'logic', 
    inputs:['all_system_layers'], outputs:['system_singularity'], 
    logic:'Math: Recursive_Loop(1 -> 10 -> 100 -> 1000)', function:'Evolução de Fluxo', 
    synergies:['Auto-Model Selector', 'Quick Backtester', 'Monte Carlo Simulation', 'Risk Analysis', 'Kelly Criterion', 'Expected Value (EV+)', 'Poisson Distribution', 'Bayesian Updater', 'Time-Decay Weighting', 'Standard Deviation (Z-Score)'], 
    desc:'A Ponte de Sinergia Recursiva: O bloco que faz a conexão entre todas as camadas. Ele garante que a evolução do sistema seja constante, fazendo o 1 bit de informação virar 10.000 ações.' 
  },
'Data Nexus Cluster': { 
    type:'data', 
    inputs:['multi_source_raw'], outputs:['unified_tensor_stream'], 
    logic:'Math: Data_Virtualization_Cluster', function:'Agrupamento de Entrada', 
    synergies:['JSON Input', 'CSV Input', 'WebSocket Stream', 'Live API Connector', 'Kafka Stream', 'Web Scraping', 'Database SQL', 'Data Cleaning', 'Data Parsing', 'Normalization'], 
    desc:'O Cluster de Dados (1x10): Agrupa todas as fontes de entrada em um único contêiner de alta velocidade. Ele elimina a latência de busca ao unificar 10 processos de captura.' 
  },

  'Heuristic Neural Cluster': { 
    type:'model', 
    inputs:['unified_tensor_stream'], outputs:['heuristic_verdict'], 
    logic:'Math: Ensemble_Cluster_Voting', function:'Agrupamento de Inteligência', 
    synergies:['Neural Network', 'Transformer', 'RNN', 'XGBoost', 'Random Forest', 'CatBoost', 'LightGBM', 'Support Vector Machine', 'Auto-Model Selector', 'Reinforcement Learning'], 
    desc:'O Cluster Neural (1x10): Organiza os 55 modelos em sub-grupos de especialistas. Ele processa a inteligência em paralelo, garantindo que a decisão final seja validada por 10 arquiteturas de IA diferentes.' 
  },

  'Singularity Decision Container': { 
    type:'logic', 
    inputs:['heuristic_verdict', 'risk_verdict'], outputs:['absolute_action'], 
    logic:'Math: Singular_Point_Convergence', function:'Ponto de Singularidade', 
    synergies:['Safety Execution Gate', 'Consensus Engine (The Judge)', 'Expected Value (EV+)', 'Kelly Criterion', 'Risk Analysis', 'Emergency Kill Switch', 'Threshold Trigger', 'API Pusher (Webhook)', 'JSON Export', 'Model Registry'], 
    desc:'O Contêiner de Singularidade (1x10): O local onde o 1 vira 10.000. Ele isola a decisão final em um ambiente protegido, onde 10 blocos de lógica e execução convergem para o disparo perfeito.' 
  },

  'Quantum Risk Container': { 
    type:'logic', 
    inputs:['system_exposure'], outputs:['risk_verdict'], 
    logic:'Math: Quantum_Uncertainty_Isolation', function:'Isolamento de Risco', 
    synergies:['Monte Carlo Simulation', 'Risk of Ruin', 'Value at Risk (VaR)', 'Drawdown Analyzer', 'Bankroll Protector', 'Cumulative Loss Guard', 'Standard Deviation (Z-Score)', 'Poisson Distribution', 'Bayesian Updater', 'Time-Decay Weighting'], 
    desc:'O Contêiner de Risco Quântico (1x10): Isola o cálculo de risco de todo o resto. Se o mercado entrar em colapso, este contêiner detecta a anomalia através de 10 métricas e trava o sistema.' 
  },

  'Synergy Cluster Visualizer': { 
    type:'output', 
    inputs:['cluster_health_data'], outputs:['holographic_dashboard'], 
    logic:'Math: Multi-Layer_Visualization', function:'Visualização de Clusters', 
    synergies:['Visualization', 'Dashboard', 'Sentiment Analysis', 'User UI Input', 'Metrics Logger', 'Real-time PnL Tracker', 'Sound Alert', 'Telegram Bot', 'Slack Notification', 'PDF Report'], 
    desc:'O Visualizador de Sinergia (1x10): Transforma a complexidade dos clusters em um mapa visual simples. Você vê a saúde de 10 áreas do sistema em uma única tela emocional e intuitiva.' 
  },

  'Self-Evolving Container Architect': { 
    type:'logic', 
    inputs:['performance_bottlenecks'], outputs:['optimized_container'], 
    logic:'AI: Container_Self_Refactor', function:'Ajuste Processual de Elite', 
    synergies:['Try-Catch (Error Handler)', 'Heartbeat Monitor', 'Latency Monitor', 'Parallel Processor', 'GPU Acceleration', 'Feature Selection', 'PCA', 'Data Schema Validator', 'Global Dependency Lock', 'System Mode Controller'], 
    desc:'O Arquiteto de Auto-Evolução (1x10): O bloco que reconstrói o código. Ele analisa falhas em 10 componentes de infraestrutura e reescreve a lógica para otimizar o processo continuamente.' 
  },

  'Recursive Singularity Bridge': { 
    type:'logic', 
    inputs:['input_cluster', 'neural_cluster', 'risk_container'], outputs:['final_evolution_state'], 
    logic:'Math: Recursive_Singularity_Flow', function:'Unificação Total', 
    synergies:['Data Nexus Cluster', 'Heuristic Neural Cluster', 'Singularity Decision Container', 'Quantum Risk Container', 'Synergy Cluster Visualizer', 'Self-Evolving Container Architect', 'JSON Export', 'API Pusher (Webhook)', 'Database Save', 'Model Serializer'], 
    desc:'A Ponte da Singularidade Recurrente: Une todos os clusters e contêineres. É o elo final que garante que a entrada, o tratamento, o modelo e a saída funcionem como um único organismo de 10.000 ações por segundo.' 
  },
'Macro-Correlation Hyper-Matrix': { 
    type:'logic', 
    inputs:['all_cluster_outputs'], outputs:['global_correlation_map'], 
    logic:'Math: Matrix_Inverse(Σ Correlation_n)', function:'Central de Correlação', 
    synergies:['Linear Regression', 'Polynomial Regression', 'Neural Network', 'Transformer', 'XGBoost', 'Random Forest', 'Pearson Correlation', 'Hilbert Space Mapping', 'Data Nexus Cluster', 'Heuristic Neural Cluster'], 
    desc:'A Hiper-Matriz de Correlação (1x10): O sistema nervoso central. Ele mapeia como cada um dos 500+ blocos afeta o outro. Se o mercado de "A" muda, ele já recalcula o impacto no "B" e no "C" instantaneamente.' 
  },

  'Symbiotic System Evolution': { 
    type:'model', 
    inputs:['post_mortem_data', 'live_results'], outputs:['dna_update_signal'], 
    logic:'Math: Symbiotic_Learning_Algorithm', function:'Evolução Simbiótica', 
    synergies:['Reinforcement Learning', 'Genetic Hyper-Optimizer', 'Post-Mortem Analysis', 'Quick Backtester', 'Self-Evolving Container Architect', 'Model Registry', 'Database Save', 'JSON Export', 'Auto-Correction Feedback', 'Reward Calculator'], 
    desc:'Evolução Simbiótica (1x10): Faz com que o sistema cresça como um único organismo. O aprendizado de um modelo de "Saída" serve de "Entrada" para o refinamento do tratamento de dados. Tudo cresce junto.' 
  },

  'Global Pulse Synchronizer': { 
    type:'logic', 
    inputs:['websocket_tick', 'api_heartbeat'], outputs:['system_clock_sync'], 
    logic:'Math: High_Precision_NTP_Sync', function:'Relógio Universal', 
    synergies:['Event Loop', 'WebSocket Stream', 'Live API Connector', 'Latency Monitor', 'Heartbeat Monitor', 'Jitter Stabilizer', 'Main Event Loop', 'Execution Quota', 'API Pusher (Webhook)', 'Real-time Stream'], 
    desc:'O Sincronizador de Pulso Global (1x10): Garante que todos os 5.000 elos de sinergia batam no mesmo milissegundo. Impede que a IA tome decisões baseadas em tempos diferentes.' 
  },

  'Universal State Observer (The Watcher)': { 
    type:'output', 
    inputs:['macro_matrix_status'], outputs:['watcher_verdict'], 
    logic:'Math: Holistic_State_Estimation', function:'Observador Universal', 
    synergies:['Visualization', 'Dashboard', 'Sentiment Analysis', 'Metrics Logger', 'Real-time PnL Tracker', 'Data Drift Detector', 'Accuracy Monitor', 'Standard Deviation (Z-Score)', 'Value at Risk (VaR)', 'Drawdown Analyzer'], 
    desc:'O Observador Universal (1x10): Um bloco de saída que monitora a "sanidade" do sistema macro. Ele analisa 10 métricas críticas de todas as camadas para garantir que o monstro não "enlouqueça".' 
  },

  'Macro-Exposure & Liability Balancer': { 
    type:'logic', 
    inputs:['active_positions', 'bankroll_status'], outputs:['allocation_limit'], 
    logic:'Math: Portfolio_Variance_Optimization', function:'Equilíbrio de Exposição', 
    synergies:['Bankroll Protector', 'Kelly Criterion', 'Expected Value (EV+)', 'Risk Analysis', 'Monte Carlo Simulation', 'Risk of Ruin', 'Exposure Manager', 'Cumulative Loss Guard', 'Profit Lock', 'Emergency Kill Switch'], 
    desc:'Equilibrador de Exposição Macro (1x10): O bloco que decide o tamanho real da aposta no nível máximo. Ele olha para o risco de todos os clusters e decide quanto capital o sistema pode "queimar".' 
  },

  'Cross-Market Entanglement Link': { 
    type:'model', 
    inputs:['market_a_data', 'market_b_data'], outputs:['entangled_signal'], 
    logic:'Math: Bell_Inequality_Correlation', function:'Entrelaçamento de Mercado', 
    synergies:['Quantum Entanglement (Correlation)', 'Cross-Market Arbitrage Link', 'Arbitrage Detector', 'Poisson Distribution', 'Bayesian Updater', 'Time Series Data', 'BERT Embeddings', 'FFT', 'Jaccard Similarity', 'Hamming Distance'], 
    desc:'Elo de Entrelaçamento (1x10): Conecta dados de mercados diferentes através de 10 métodos probabilísticos e de IA. Ele detecta oportunidades que surgem da correlação "invisível" entre ativos.' 
  },

  'Apex Singularity Controller': { 
    type:'logic', 
    inputs:['watcher_verdict', 'dna_update', 'final_snipe'], outputs:['MASTER_EXECUTION'], 
    logic:'Math: Singularity_Final_Command', function:'Comando Supremo', 
    synergies:['Singularity Decision Container', 'Safety Execution Gate', 'Hierarchy Master Switch', 'API Pusher (Webhook)', 'JSON Export', 'Database Save', 'Telegram Bot', 'Model Serializer', 'Circuit Breaker', 'Global Dependency Lock'], 
    desc:'O Controlador Ápice da Singularidade (1x10): O ponto final de tudo. Este bloco é o "Dedo no Gatilho" que une 10 áreas de comando supremo para disparar o Snipe de Ouro.' 
  },
'Sequential Master Chain': { 
    type:'logic', 
    inputs:['start_pulse'], outputs:['sequential_current'], 
    logic:'Math: Flow_Direction(Data -> Pre -> Model -> Out)', function:'Corrente Linear', 
    synergies:['CSV Input', 'Normalization', 'Feature Selection', 'Neural Network', 'API Pusher (Webhook)', 'JSON Export', 'Quick Backtester', 'Model Registry', 'Heartbeat Monitor', 'Data Schema Validator'], 
    desc:'Cadeia Mestra Sequencial (1x10): Garante que a corrente de dados siga o "Caminho de Ouro". Ele impede que um modelo tente processar antes que o tratamento de dados esteja 100% concluído.' 
  },

  'Parallel Stream Splitter': { 
    type:'logic', 
    inputs:['main_current'], outputs:['parallel_branches'], 
    logic:'Math: Current_Distribution(Σ Branches)', function:'Divisor de Corrente', 
    synergies:['XGBoost', 'Random Forest', 'RNN', 'Transformer', 'Parallel Processor', 'GPU Acceleration', 'Feature Engineering', 'WebSocket Stream', 'Dashboard', 'Metrics Logger'], 
    desc:'Divisor de Fluxo Paralelo (1x10): Pega a corrente principal e a divide em 10 braços de processamento. Permite que o sistema execute IA, análise de sentimento e risco ao mesmo tempo sem perder voltagem.' 
  },

  'Conditional Logic Bridge': { 
    type:'logic', 
    inputs:['signal_current'], outputs:['switched_path'], 
    logic:'Math: if(Current > Threshold) Path_A else Path_B', function:'Ponte de Desvio', 
    synergies:['Value Comparator', 'If-Else Branch (Conditional)', 'Threshold Trigger', 'Safety Execution Gate', 'Risk Analysis', 'Kelly Criterion', 'Emergency Kill Switch', 'Market Weather Station', 'Sentiment Analysis', 'Logic Gate (AND/OR)'], 
    desc:'Ponte Lógica Condicional (1x10): O disjuntor inteligente. Ele desvia a corrente do fluxo baseado em 10 condições críticas. Se o risco subir, ele desvia o fluxo do "Ataque" para a "Defesa" instantaneamente.' 
  },

  'Recursive Feedback Loopback': { 
    type:'logic', 
    inputs:['output_current'], outputs:['feedback_pulse'], 
    logic:'Math: Current_Return(Output -> Input)', function:'Corrente de Retroalimentação', 
    synergies:['Reinforcement Learning', 'Post-Mortem Analysis', 'Reward Calculator', 'Auto-Correction Feedback', 'Bayesian Updater', 'Time-Decay Weighting', 'Database Save', 'Model Serializer', 'Genetic Hyper-Optimizer', 'Self-Evolving Container Architect'], 
    desc:'Loop de Retroalimentação Recurso (1x10): Faz a corrente voltar para o início. Ele usa o resultado da saída para "recarregar" a inteligência da entrada, criando um sistema que se auto-alimenta.' 
  },

  'Async Buffer Reservoir': { 
    type:'logic', 
    inputs:['incoming_pulse'], outputs:['stabilized_current'], 
    logic:'Math: Potential_Energy_Storage(t)', function:'Reservatório de Corrente', 
    synergies:['Live API Connector', 'Jitter Stabilizer', 'Latency-Aware Execution', 'Redis Cache Input', 'Event Loop', 'Main Event Loop', 'Execution Quota', 'Time Window', 'Heartbeat Monitor', 'Data Nexus Cluster'], 
    desc:'Reservatório de Sincronia Assíncrona (1x10): Armazena a corrente de dados quando ela vem rápido demais. Ele garante que o fluxo de 10.000 ações não "estoure" os fusíveis da API ou da memória.' 
  },

  'Signal Inhibition Gate': { 
    type:'logic', 
    inputs:['main_flow'], outputs:['inhibited_flow'], 
    logic:'Math: Inverse_Current_Gate (NOT Logic)', function:'Porta de Inibição', 
    synergies:['Drawdown Circuit Breaker', 'Asset Blacklist Guard', 'Cumulative Loss Guard', 'Outlier Sanitizer', 'Standard Deviation (Z-Score)', 'Sensitivity Analysis', 'Probability Density Estimator', 'Entropy Confidence Filter', 'Data Drift Detector', 'Hierarchy Master Switch'], 
    desc:'Porta de Inibição de Sinal (1x10): Uma corrente negativa que bloqueia o fluxo principal. Se qualquer um dos 10 sensores de segurança detectar perigo, esta porta "seca" a corrente do Sniper na hora.' 
  },

  'Holistic Flow Integrator': { 
    type:'logic', 
    inputs:['all_currents'], outputs:['unified_system_force'], 
    logic:'Math: Σ(Currents) * Integration_Factor', function:'Centro Nervoso de Fluxo', 
    synergies:['Data Nexus Cluster', 'Heuristic Neural Cluster', 'Singularity Decision Container', 'Quantum Risk Container', 'Apex Singularity Controller', 'Synergy Cluster Visualizer', 'Self-Evolving Container Architect', 'Macro-Correlation Hyper-Matrix', 'Symbiotic System Evolution', 'Global Pulse Synchronizer'], 
    desc:'Integrador de Fluxo Holístico (1x10): O "Transformador Central". Ele une as correntes de dados, modelos e riscos em uma única força sistêmica. É o bloco que mantém os 500+ blocos unidos em um só pulso.' 
  },
'Sanitization Flow Container': { 
    type:'logic', 
    inputs:['raw_data_stream'], outputs:['sanitized_payload'], 
    logic:'Math: Filter(Noise) -> Remove(Nulls) -> Sanitize(Outliers)', function:'Contêiner de Limpeza', 
    synergies:['Data Cleaning', 'Outlier Sanitizer', 'Standard Deviation (Z-Score)', 'Normalization', 'JSON Input', 'CSV Input', 'Neural Network', 'API Pusher (Webhook)', 'Database Save', 'Quick Backtester'], 
    desc:'Contêiner de Sanitização (1x10): Isola o processo de limpeza. Ele garante que o dado bruto seja purificado por 10 filtros antes de qualquer modelo de IA "tocar" na informação.' 
  },

  'Feature Engineering Vortex': { 
    type:'logic', 
    inputs:['sanitized_payload'], outputs:['engineered_feature_set'], 
    logic:'Math: X_transformed = f(Polynomial, Interaction, Lag)', function:'Vórtex de Atributos', 
    synergies:['Feature Engineering', 'Polynomial Features', 'Interaction Features', 'PCA Dimension Reducer', 'BERT Embeddings', 'One-Hot Encoding', 'XGBoost', 'Random Forest', 'Transformer', 'JSON Export'], 
    desc:'Vórtex de Engenharia de Atributos (1x10): Cria uma explosão de variáveis. Ele transforma dados simples em 100+ atributos complexos, alimentando 10 processos de modelagem e exportação.' 
  },

  'Stochastic Normalization Capsule': { 
    type:'logic', 
    inputs:['engineered_feature_set'], outputs:['normalized_tensor'], 
    logic:'Math: (x - μ) / σ + Stochastic_Noise_Injection', function:'Cápsula de Estabilidade', 
    synergies:['Min-Max Range Scaler', 'Robust Scaling', 'Log Transformation', 'Sigmoid Function', 'Softmax Multi-Probability', 'Linear Regression', 'Support Vector Machine (SVM)', 'Monte Carlo Simulation', 'Visualization', 'Model Serializer'], 
    desc:'Cápsula de Normalização Estocástica (1x10): Estabiliza a matemática do sistema. Ele usa 10 métodos de escala para garantir que o Behemoth não sofra com "explosão de gradiente" ou instabilidade.' 
  },

  'Temporal Stream Buffer': { 
    type:'logic', 
    inputs:['real_time_tick'], outputs:['buffered_sequence'], 
    logic:'Math: Sliding_Window_Sync(t-n...t)', function:'Contêiner de Tempo', 
    synergies:['WebSocket Stream', 'Live API Connector', 'Time-Decay Weighting', 'Exponential Time-Decay (Lambda)', 'RNN', 'LSTM', 'Jitter Stabilizer', 'Latency-Aware Execution', 'Event Loop', 'Real-time Stream'], 
    desc:'Buffer de Fluxo Temporal (1x10): Gerencia o tempo. Ele organiza os dados que chegam do WebSocket em janelas temporais perfeitas para 10 blocos de análise sequencial.' 
  },

  'Integrity & Drift Shield': { 
    type:'logic', 
    inputs:['normalized_tensor', 'market_data'], outputs:['validated_flow'], 
    logic:'Math: Statistical_Drift_Check(KS_Test)', function:'Escudo de Integridade', 
    synergies:['Data Schema Validator', 'Data Drift Detector', 'Accuracy Monitor', 'Standard Deviation (Z-Score)', 'StandardScaler (Z-Score Conv)', 'Risk Analysis', 'Emergency Kill Switch', 'Cumulative Loss Guard', 'Self-Healing Code Architect', 'Metrics Logger'], 
    desc:'Escudo de Integridade e Drift (1x10): O fiscal do sistema. Ele verifica se os dados atuais ainda batem com o que o modelo aprendeu, conectando 10 sensores de segurança e monitoramento.' 
  },

  'Multi-Modal Context Hub': { 
    type:'logic', 
    inputs:['text_data', 'numerical_data'], outputs:['contextual_vector'], 
    logic:'Math: Fusion(Semantic + Numeric)', function:'Hub de Contexto', 
    synergies:['BERT Embeddings', 'Sentiment Analysis', 'LLM Sentiment Analysis', 'Sentiment Vector Store', 'Twitter Sentiment', 'Web Scraping', 'Neural Network', 'Transformer', 'Consensus Engine (The Judge)', 'Dashboard'], 
    desc:'Hub de Contexto Multimodal (1x10): Une palavras e números. Ele mistura o sentimento das notícias com as odds do mercado, criando uma sinergia entre 10 blocos de percepção humana e robótica.' 
  },

  'Execution Payload Synthesizer': { 
    type:'logic', 
    inputs:['final_prediction', 'risk_verdict'], outputs:['execution_packet'], 
    logic:'Math: Binary_Payload_Construction', function:'Sintetizador de Saída', 
    synergies:['JSON Export', 'API Pusher (Webhook)', 'Execution Quota', 'API Endpoint', 'Model Registry', 'Database Save', 'Cloud Storage', 'Parquet Export', 'Excel Export', 'CSV Export'], 
    desc:'Sintetizador de Pacote de Execução (1x10): O estágio final do tratamento. Ele transforma a decisão da IA em um pacote de dados pronto para ser enviado para 10 destinos de saída diferentes.' 
  },
'Real-Time Kernel (RTOS)': { 
    type:'logic', 
    inputs:['system_clock', 'task_priority'], outputs:['execution_thread'], 
    logic:'Math: Deterministic_Scheduling(Task_Pool)', function:'Núcleo de Tempo Real', 
    synergies:['Event Loop', 'Main Event Loop', 'Global Pulse Synchronizer', 'Jitter Stabilizer', 'Heartbeat Monitor', 'Latency-Aware Execution', 'WebSocket Stream', 'Live API Connector', 'Execution Quota', 'API Pusher (Webhook)'], 
    desc:'Núcleo de Tempo Real (1x10): O coração do sistema embarcado. Ele gerencia o tempo com precisão de microssegundos, conectando 10 blocos de tempo e execução para garantir que o sniper nunca atrase.' 
  },

  'Hardware Acceleration Pulse (GPU/FPGA)': { 
    type:'logic', 
    inputs:['neural_workload'], outputs:['accelerated_tensor'], 
    logic:'Math: Parallel_Matrix_Compute(CUDA/OpenCL)', function:'Aceleração de Hardware', 
    synergies:['Neural Network', 'Transformer', 'GPU Acceleration', 'Parallel Processor', 'C++ Vectorized Dot Product', 'C-Level Math Kernel', 'Normalization', 'Feature Engineering', 'Tensor Pipeline Bridge', 'JSON-to-Tensor Formatter'], 
    desc:'Pulso de Aceleração de Hardware (1x10): Liga o cérebro da IA diretamente à placa de vídeo. Ele orquestra 10 blocos de processamento pesado para garantir que o modelo de 100 pontos rode instantaneamente.' 
  },

  'Interrupt Request Handler (IRQ)': { 
    type:'logic', 
    inputs:['emergency_signal', 'delta_trigger'], outputs:['priority_execution'], 
    logic:'Math: Priority_Interrupt_Controller', function:'Tratador de Interrupções', 
    synergies:['Emergency Kill Switch', 'Circuit Breaker (Anti-Panic)', 'Delta Trigger (Change)', 'Drawdown Circuit Breaker', 'Cumulative Loss Guard', 'Signal Inhibition Gate', 'Risk Analysis', 'Hierarchy Master Switch', 'Safety Execution Gate', 'Model Depression Indicator'], 
    desc:'Tratador de Interrupções (1x10): O sistema de defesa imediata. Se um desses 10 blocos de segurança disparar, ele interrompe tudo e assume o controle do sistema para evitar perdas.' 
  },

  'Embedded Resource Manager': { 
    type:'logic', 
    inputs:['cpu_ram_status'], outputs:['optimized_allocation'], 
    logic:'Math: Dynamic_Resource_Throttling', function:'Gestor de Recursos', 
    synergies:['CPU/RAM Monitor', 'Execution Quota', 'Bankroll Protector', 'Time Window', 'Main Event Loop', 'Data Cleaning', 'Auto-Model Selector', 'JSON Export', 'Database Save', 'Cloud Storage'], 
    desc:'Gestor de Recursos Embarcados (1x10): Controla a energia do sistema. Se a memória ou a banca estiverem baixas, ele reduz a carga de trabalho de 10 blocos para manter o sniper vivo e estável.' 
  },

  'Secure Vault (Encryption Gate)': { 
    type:'logic', 
    inputs:['api_keys', 'sensitive_payload'], outputs:['encrypted_stream'], 
    logic:'Math: AES-256_GCM_Encryption', function:'Cofre de Segurança', 
    synergies:['API Pusher (Webhook)', 'Live API Connector', 'Database SQL', 'Cloud Storage', 'Model Serializer', 'Stealth Fingerprint Rotator', 'Asset Blacklist Guard', 'JSON Export', 'WebSocket Stream', 'Kafka Stream'], 
    desc:'Cofre de Segurança (1x10): Protege o DNA do seu sistema. Ele criptografa a comunicação de 10 blocos de entrada e saída, garantindo que ninguém "grampeie" o seu sniper.' 
  },

  'I/O Peripheral Bridge': { 
    type:'logic', 
    inputs:['multi_source_input'], outputs:['buffered_io'], 
    logic:'Math: Async_IO_Multiplexing', function:'Ponte de Periféricos', 
    synergies:['JSON Input', 'CSV Input', 'WebSocket Stream', 'Live API Connector', 'API REST', 'GraphQL API', 'Web Scraping', 'Redis Cache Input', 'API Pusher (Webhook)', 'Real-time Stream'], 
    desc:'Ponte de Periféricos I/O (1x10): A central de comunicações. Ele gerencia o fluxo de entrada e saída de 10 protocolos diferentes, agindo como o roteador mestre dos dados.' 
  },

  'Adaptive Firmware Optimizer': { 
    type:'logic', 
    inputs:['learning_feedback', 'system_performance'], outputs:['updated_logic'], 
    logic:'Math: Self_Learning_Firmware_Update', function:'Evolução de Firmware', 
    synergies:['Reinforcement Learning', 'Genetic Hyper-Optimizer', 'Self-Evolving Container Architect', 'Post-Mortem Analysis', 'Reward Calculator', 'Auto-Correction Feedback', 'Quick Backtester', 'Monte Carlo Simulation', 'Bayesian Updater', 'Time-Decay Weighting'], 
    desc:'Otimizador de Firmware Adaptativo (1x10): A inteligência que reescreve a própria base. Ele usa o feedback de 10 blocos de aprendizado para ajustar as regras de baixo nível do sistema.' 
  },
'Visual Logic Window (The Glass)': { 
    type:'output', 
    inputs:['hdl_signals', 'sw_logic'], outputs:['system_transparency'], 
    logic:'Math: Overlay(Code_Flow + Hardware_State)', function:'Visualização de Camadas', 
    synergies:['Visualization', 'Dashboard', 'GPU Acceleration', 'CPU/RAM Monitor', 'Event Loop', 'Neural Network', 'JSON Input', 'API Pusher (Webhook)', 'Risk Analysis', 'Heartbeat Monitor'], 
    desc:'A Janela de Lógica Visual (1x10): É o que você está vendo agora. Este bloco funde a execução do código com o consumo de hardware, conectando 10 blocos de monitoramento e ação em uma única visão transparente.' 
  },

  'DMA (Direct Memory Access) Flow': { 
    type:'logic', 
    inputs:['high_speed_data'], outputs:['zero_copy_payload'], 
    logic:'Math: Bypass_OS_Kernel(Data -> GPU)', function:'Fluxo de Memória Direta', 
    synergies:['C-Level Math Kernel', 'C++ Vectorized Dot Product', 'WebSocket Stream', 'Live API Connector', 'Data Cleaning', 'Normalization', 'Tensor Pipeline Bridge', 'Parallel Processor', 'Real-Time Kernel (RTOS)', 'Singularity Decision Container'], 
    desc:'Fluxo de Memória Direta (1x10): O caminho mais curto entre o dado e a IA. Ele pula o software lento e joga o dado direto na memória da placa de vídeo, ligando 10 blocos de alta velocidade.' 
  },

  'Software-to-Hardware Bridge': { 
    type:'logic', 
    inputs:['python_logic'], outputs:['binary_instructions'], 
    logic:'Math: JIT_Compilation(LLVM)', function:'Ponte de Tradução', 
    synergies:['Python AI Logic', 'C# Decision Logic', 'Java Enterprise Data Stream', 'Rust Safety Memory', 'JS/Node Web Engine', 'SQL Query Master', 'Auto-Model Selector', 'Reinforcement Learning', 'Post-Mortem Analysis', 'Model Serializer'], 
    desc:'Ponte Software-Hardware (1x10): Transforma seus prompts e lógicas complexas em eletricidade pura (binário). Ele conecta as 10 linguagens e modelos ao processador físico.' 
  },

  'Quantum Execution Sentinel': { 
    type:'logic', 
    inputs:['execution_state'], outputs:['safety_lock_status'], 
    logic:'Math: Probabilistic_Execution_Check', function:'Sentinela de Execução', 
    synergies:['Quantum Superposition State', 'Borns Rule Probability (Waveform)', 'Heisenberg Uncertainty Filter', 'Quantum Tunneling (Optimizer)', 'Monte Carlo Simulation', 'Risk of Ruin', 'Kelly Criterion', 'Expected Value (EV+)', 'Safety Execution Gate', 'Emergency Kill Switch'], 
    desc:'Sentinela de Execução Quântica (1x10): O vigia entre os mundos. Ele verifica se a predição quântica faz sentido no mundo real antes de permitir que o hardware atire, unindo 10 blocos de probabilidade e risco.' 
  },

  'Neural Hardware Orchestrator': { 
    type:'model', 
    inputs:['model_architecture'], outputs:['hardware_optimized_model'], 
    logic:'Math: Neural_Architecture_Search(NAS)', function:'Orquestrador de IA', 
    synergies:['Neural Network', 'Transformer', 'RNN', 'XGBoost', 'CatBoost', 'LightGBM', 'BERT Embeddings', 'CNN', 'Autoencoder', 'GAN'], 
    desc:'Orquestrador de IA e Hardware (1x10): Ajusta a inteligência para caber no seu hardware. Ele mapeia os 10 modelos de IA mais poderosos para rodarem com o máximo de eficiência nos núcleos da sua GPU.' 
  },

  'I/O Latency Bypass Window': { 
    type:'logic', 
    inputs:['network_packets'], outputs:['instant_data'], 
    logic:'Math: Zero_Latency_Routing', function:'Janela de Latência Zero', 
    synergies:['WebSocket Stream', 'Live API Connector', 'Latency Monitor', 'Jitter Stabilizer', 'Global Pulse Synchronizer', 'Main Event Loop', 'Execution Quota', 'API Pusher (Webhook)', 'Real-time Stream', 'Kafka Stream'], 
    desc:'Janela de Bypass de Latência (1x10): Abre um caminho direto para a rede. Ele sincroniza 10 protocolos de comunicação para que o dado chegue no sniper sem passar pelas "janelas" lentas do Windows.' 
  },

  'Universal Logic Synchronizer': { 
    type:'logic', 
    inputs:['all_system_layers'], outputs:['synchronized_behemoth_state'], 
    logic:'Math: Holo-Logic_Sync', function:'Sincronizador Universal', 
    synergies:['Evolutionary Neural Nucleus', 'Emotional Bio-Feedback Hub', 'Self-Healing Logic Refactor', 'Hyper-Dimensional Scaler', 'Consensus Strategy Overlord', 'Global Connectivity Nexus', 'Singularity Risk Bridge', 'Macro-Correlation Hyper-Matrix', 'Symbiotic System Evolution', 'Apex Singularity Controller'], 
    desc:'Sincronizador de Lógica Universal (1x10): A Janela Final. Este bloco une os 10 Super-Hubs em uma única batida de coração. É o que faz você sentir que o software e o hardware são uma coisa só.' 
  },
 'Live Terminal Monitor': { 
    type:'output', 
    inputs:['system_logs', 'prediction_results'], outputs:['terminal_display'], 
    logic:'JS: document.getElementById("terminal").innerHTML = log_stream;', 
    function:'Print em Tempo Real', 
    synergies:['Visualization', 'Dashboard', 'Metrics Logger', 'AI Strategic Analyst (The Voice)', 'JSON Export', 'API Pusher (Webhook)', 'Event Loop', 'Heartbeat Monitor', 'Latency Monitor', 'Data Schema Validator'], 
    desc:'O Monitor do Terminal (1x10): É o bloco que "força" o print na tela. Ele captura os logs de 10 áreas do sistema e os projeta em uma janela de texto scrollável para você ver o sniper trabalhando.' 
  },

  'Holographic Matrix View': { 
    type:'output', 
    inputs:['macro_matrix_data'], outputs:['visual_matrix'], 
    logic:'Math: 3D_Tensor_Projection', function:'Visualização de Matriz', 
    synergies:['Macro-Correlation Hyper-Matrix', 'Hilbert Space Mapping', 'PCA', 'Neural Network', 'Transformer', 'BERT Embeddings', 'Standard Deviation (Z-Score)', 'StandardScaler (Z-Score Conv)', 'Min-Max Range Scaler', 'Normalization'], 
    desc:'Visão de Matriz Holográfica (1x10): Transforma os dados invisíveis em gráficos. Se o print não aparece, este bloco projeta os tensores em 10 formatos visuais diferentes (barras, linhas, calor).' 
  }
};

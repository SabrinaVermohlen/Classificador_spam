# Classificador de Spam em SMS

Projeto desenvolvido para classificar mensagens de SMS como "spam" ou "ham" (não spam) usando técnicas de Machine Learning. A abordagem principal foi a utilização de Máquinas de Vetores de Suporte (SVM) junto com vetorização de texto.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

## Sobre o projeto

O objetivo foi construir um modelo capaz de identificar se uma mensagem recebida é spam ou não. Para isso, foi usado o dataset "SMSSpamCollection", que contém mensagens já rotuladas.

O processo inclui:
- Carregamento e análise exploratória dos dados
- Visualização da distribuição de mensagens spam e ham
- Pré-processamento com transformação dos textos usando CountVectorizer
- Construção de um pipeline de Machine Learning combinando vetorização e SVM
- Otimização dos hiperparâmetros através de GridSearchCV

Além disso, foram analisados os erros cometidos pelo modelo, como falsos positivos e falsos negativos.

## Resultados finais

Após a otimização, o modelo apresentou bons resultados de classificação, avaliados através de métricas como acurácia, precisão, recall e F1-score. A matriz de confusão também foi gerada para identificar os principais erros de classificação.

Algumas métricas observadas:
- Acurácia média nos testes
- Precisão na detecção de mensagens de spam
- Recall mostrando o quanto o modelo foi capaz de identificar corretamente os spams

Esses resultados mostram que a abordagem utilizando SVM foi bastante eficiente para o problema proposto.

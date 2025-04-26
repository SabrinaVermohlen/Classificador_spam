# 📩 Classificador de Spam em SMS

Projeto criado para classificar mensagens de SMS como "spam" ou "ham" (mensagens legítimas), utilizando técnicas de Machine Learning. A principal abordagem foi o uso de SVM (Máquinas de Vetores de Suporte) combinada com vetorização de texto.

## 🚀 Tecnologias utilizadas

• Python  
• Pandas  
• Scikit-learn  
• Seaborn  
• Matplotlib

## ✨ Sobre o projeto

A ideia foi construir um modelo capaz de identificar se uma mensagem é spam ou não. Para isso, foi utilizado o dataset "SMSSpamCollection", que contém mensagens rotuladas como "spam" ou "ham".

As etapas desenvolvidas foram:
• Análise exploratória e visualização da distribuição entre spam e ham  
• Pré-processamento dos textos usando CountVectorizer  
• Criação de um pipeline combinando a vetorização e o modelo SVM  
• Otimização de hiperparâmetros com GridSearchCV  
• Análise dos erros mais comuns, como falsos positivos e falsos negativos

## 📊 Resultados finais

Após a otimização, o modelo obteve resultados bem consistentes, avaliados pelas principais métricas de classificação:

• Acurácia geral do modelo  
• Precisão para identificar mensagens de spam  
• Recall para medir o quanto de spam foi corretamente detectado  
• F1-Score equilibrando precisão e recall  
• Matriz de confusão para análise detalhada dos erros

Esses resultados mostram que a abordagem escolhida foi eficiente para o problema proposto, com bom desempenho na detecção de mensagens indesejadas.

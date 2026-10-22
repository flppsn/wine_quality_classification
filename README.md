# Classificação da qualidade de vinhos com Machine Learning

Projeto do Tech Challenge da Fase 2 da Pós-Graduação em Data Analytics (POSTECH).

O objetivo é prever se um vinho é de **alta qualidade (nota ≥ 7)** ou de
**baixa/média qualidade (nota < 7)** a partir de suas características
físico-químicas, como acidez, teor alcoólico, densidade e sulfatos.

## Contexto

A qualidade de um vinho é tradicionalmente avaliada por especialistas, em
análises sensoriais que podem ser subjetivas e demoradas. Um modelo preditivo
baseado em dados de laboratório pode apoiar enólogos e produtores, indicando
quais lotes têm maior chance de serem de alta qualidade.

## Base de dados

- **Fonte:** Wine Quality Dataset (Kaggle)
- **Tamanho:** 1.143 amostras, 11 variáveis físico-químicas e a nota de qualidade
- **Após remover duplicatas:** 1.018 vinhos, sendo 137 de alta qualidade (13,5%)

## Estrutura do repositório

```
├── data/
│   └── Base.csv                      # Base de dados utilizada
├── notebooks/
│   └── 01_analise_vinhos.ipynb       # Análise completa: EDA, modelagem e interpretação
├── src/
│   └── treinar_modelo.py             # Pipeline do modelo final (preparação, treino e métricas)
├── results/                          # Gráficos e métricas dos modelos
├── apresentacao/                     # Apresentação executiva (storytelling da EDA)
├── requirements.txt                  # Bibliotecas utilizadas
└── README.md
```

## Como executar

Requer Python 3.13 (versão utilizada no desenvolvimento).

1. Instale as bibliotecas:

   ```
   pip install -r requirements.txt
   ```

2. Para ver a análise completa, abra o notebook `notebooks/01_analise_vinhos.ipynb`
   e execute todas as células em ordem.

3. Para treinar apenas o modelo final e gerar as métricas, execute a partir da
   pasta raiz do projeto:

   ```
   python src/treinar_modelo.py
   ```

## Metodologia

1. **Compreensão do problema:** criação da variável alvo binária
   `qualidade_alta` (1 = nota ≥ 7).
2. **Análise exploratória:** distribuição das variáveis, comparação entre as
   classes, outliers, correlações e duplicatas.
3. **Pré-processamento:**
   - remoção de 125 linhas duplicadas;
   - transformação logarítmica nas variáveis com cauda longa, em vez de remover
     outliers (que atingiriam 25,8% da base);
   - criação de duas variáveis: `sulfatos_por_cloretos` e `razao_so2_livre`;
   - divisão estratificada em treino (80%) e teste (20%);
   - padronização ajustada somente com os dados de treino.
4. **Modelos:** Regressão Logística, KNN e Random Forest, comparados com um
   modelo de referência que sempre prevê "baixa/média qualidade".
5. **Avaliação:** precisão, recall, F0,5 e AUC, com validação cruzada de
   5 partes no treino e avaliação final no teste.
6. **Interpretação:** pesos da Regressão Logística e importância das variáveis
   na Random Forest.

### Critério de avaliação

O erro mais grave para a vinícola é **classificar um vinho comum como premium**,
pois compromete a reputação da marca. Por isso, a métrica principal é o
**F0,5**, que dá mais peso à precisão sem ignorar o recall.

A acurácia não foi usada como critério: como 86% dos vinhos são de
baixa/média qualidade, um modelo que nunca identifica um vinho premium já
acertaria 86%.

## Resultados

**Modelo escolhido:** Regressão Logística, classificando como alta qualidade os
vinhos com probabilidade a partir de 45%. Teve o melhor F0,5 e a melhor AUC na
validação cruzada, além de ser interpretável.

| Métrica (conjunto de teste) | Valor |
|---|---|
| Precisão | 0,636 |
| Recall | 0,519 |
| F0,5 | 0,609 |
| AUC | 0,886 |

- De 22 vinhos indicados pelo modelo como de alta qualidade, 14 de fato eram.
- Sem o modelo, apenas 13% dos vinhos são de alta qualidade. Entre os indicados
  pelo modelo, 64% são.
- O modelo identifica cerca de metade dos vinhos de alta qualidade (14 de 27).

## Principais conclusões

- **Teor alcoólico** é a variável mais associada a vinhos de alta qualidade.
- **Acidez volátil** é a principal variável associada à baixa qualidade. Em
  excesso, ela confere sabor avinagrado.
- **Sulfatos** também se associam positivamente à qualidade.
- Essas três conclusões foram confirmadas pela análise exploratória e pelos
  dois modelos.
- As variáveis criadas não alteraram o desempenho de forma relevante.

## Implicações para a produção

- Usar o modelo como **triagem**: priorizar na avaliação sensorial os lotes
  que o modelo indica como de alta qualidade.
- Monitorar o teor alcoólico e a acidez volátil como indicadores de qualidade.
- Por se tratar de associações, e não de relações de causa e efeito, mudanças
  no processo devem ser testadas em lotes controlados antes de serem adotadas.

## Limitações

- Base pequena (137 vinhos de alta qualidade), o que torna as métricas
  sensíveis a poucos acertos ou erros.
- A nota de qualidade vem de avaliações sensoriais, com componente subjetivo.
- A base não inclui fatores como tipo de uva, safra e região.

## Autores

Evandro Filippsen e Lavínia Honório

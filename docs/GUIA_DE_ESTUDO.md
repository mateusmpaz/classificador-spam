# Entender e apresentar o projeto

## O que foi construído

Uma aplicação Python de classificação binária de textos, com aprendizado supervisionado. A base traz exemplos com respostas conhecidas: `ham` ou `spam`. O treinamento usa essas respostas para ajustar um modelo. Na inferência, o modelo recebe apenas um texto novo e produz uma previsão.

O rótulo não é descoberto pelo `print`. Ele entra como resposta conhecida durante o treinamento e como referência para medir erros na avaliação. Ao classificar uma mensagem digitada, não entregamos seu rótulo ao modelo.

## Percurso do programa

1. **Leitura:** `carregar_mensagens` interpreta o arquivo e rejeita registros inválidos. `split("\t", 1)` separa apenas a primeira tabulação; as demais pertencem ao texto.
2. **Inspeção:** `contar_rotulos` mostra o desbalanceamento. A maioria das mensagens é ham.
3. **Limpeza:** `remover_duplicatas` mantém o primeiro texto idêntico e rejeita rótulos conflitantes. Ela roda antes da divisão para evitar que cópias exatas apareçam nos dois lados da avaliação.
4. **Divisão:** `dividir_dados` reserva 20% para teste. Dos 80% restantes, reserva 25% para validação: 0,25 × 0,80 = 20% do total. Sobra aproximadamente 60% para treino.
5. **Representação:** TF-IDF transforma textos em vetores numéricos. Cada coluna representa um termo do vocabulário aprendido no treino. Os pesos combinam frequência no documento e raridade no conjunto usado no ajuste; a configuração padrão também normaliza os vetores.
6. **Aprendizado:** Naive Bayes e regressão logística usam esses vetores e os rótulos para ajustar seus parâmetros.
7. **Comparação:** as previsões de validação são comparadas com os rótulos reais. F1 de spam é o critério usado para escolher entre os dois modelos.
8. **Treino final:** após escolher a regressão logística, um novo vetorizador e um novo modelo aprendem com treino + validação.
9. **Teste:** a opção `--avaliar-teste` aplica o modelo final às mensagens reservadas. O teste não participa da escolha do modelo.
10. **Uso:** o terminal transforma a mensagem digitada com o vetorizador final e pede a previsão ao modelo final.

## Por que `fit`, `transform` e `predict` são diferentes

| Operação | O que acontece |
|---|---|
| `vetorizador.fit_transform(textos_treino)` | Aprende vocabulário e pesos IDF; transforma o treino em números. |
| `vetorizador.transform(textos_validacao)` | Usa a representação já aprendida, sem ajustar o vocabulário. |
| `modelo.fit(X_treino, rotulos_treino)` | Ajusta parâmetros a partir de exemplos e respostas. |
| `modelo.predict(X_validacao)` | Produz previsões sem receber as respostas reais. |

Usar `fit_transform` na validação criaria outra representação e permitiria que a avaliação influenciasse o ajuste. Aprender o vocabulário em toda a base antes da divisão também introduziria informação do conjunto reservado.

No treino final, usar treino + validação é intencional: a escolha já foi feita. O teste continua separado.

## O que os modelos aprendem

**Multinomial Naive Bayes:** combina evidências dos termos para estimar a classe, com uma hipótese simplificadora de independência condicional entre características dada a classe. Aqui ele recebe os pesos não negativos do TF-IDF. Não é uma lista manual de palavras proibidas.

**Regressão logística:** aprende pesos associados às características e um intercepto para separar as classes. Embora o nome contenha “regressão”, aqui é um classificador. A configuração usada aplica regularização; não fizemos busca de hiperparâmetros nem ajuste do limiar.

**TF-IDF:** não entende todo o significado da frase. Na configuração usada, trabalha com termos individuais, converte maiúsculas em minúsculas e ignora termos fora do vocabulário. Uma mensagem com vetor inteiramente zero recebe “sem classificação” na interação.

## Entender as métricas

Considerando spam como a classe positiva:

- **Verdadeiro positivo (TP):** spam previsto como spam.
- **Falso positivo (FP):** ham previsto como spam; pode esconder uma mensagem legítima.
- **Falso negativo (FN):** spam previsto como ham; deixa passar uma mensagem indesejada.
- **Verdadeiro negativo (TN):** ham previsto como ham.

```text
Acurácia = (TP + TN) / total
Precisão = TP / (TP + FP)
Recall   = TP / (TP + FN)
F1       = 2 × precisão × recall / (precisão + recall)
```

Na validação da regressão logística: TP = 97, FP = 1, FN = 33, TN = 903. Precisão responde “dos que marquei como spam, quantos eram spam?”. Recall responde “dos spams existentes, quantos encontrei?”. F1 combina as duas por média harmônica.

A referência sempre ham acerta 87,43% da validação e não detecta nenhum spam. Isso mostra por que acurácia isolada é inadequada para decidir aqui. Os 86,60% vistos no começo pertenciam à base original, antes da limpeza e da divisão: não são a referência correta para comparar modelos na validação.

F1 é uma escolha inicial de avaliação, não uma regra universal. Em uma aplicação real, o custo de perder mensagens legítimas pode exigir outro critério, como recall sujeito a uma precisão mínima.

## Detalhes que um avaliador pode procurar

| Pergunta | Evidência neste projeto |
|---|---|
| A base tem origem e licença? | Atribuição, DOI, documentação original e hash em `dados/`. |
| Existem duplicatas ou rótulos conflitantes? | Inspeção e tratamento antes de dividir os dados. |
| O modelo apenas aproveita a classe majoritária? | Comparação com a referência sempre ham. |
| Houve vazamento de dados? | Ajuste do TF-IDF limitado ao treino; teste fora da seleção. Ainda há risco de mensagens semelhantes. |
| A avaliação considera os erros da classe minoritária? | Precisão, recall, F1 de spam e matriz de confusão. |
| A escolha ocorreu antes do teste? | Regressão logística escolhida pelo F1 de validação; teste opcional separado. |
| É possível repetir a execução? | Base incluída, seed fixa, dependência especificada, comandos e relatório. |
| As limitações são reconhecidas? | Escopo em SMS inglês, uma divisão, ausência de avaliação temporal e de produção. |
| O código lida com entradas problemáticas? | Validação do arquivo, interrupção em conflitos e tratamento de textos sem termos. |

## Mudanças de organização em relação ao arquivo de aula

O experimento continua usando os mesmos modelos e divisões. O pacote extraiu blocos para funções, adicionou testes, uma referência na validação, opções de terminal e exportação das métricas em JSON.

- `argparse` lê opções como `--avaliar-teste`.
- `action="store_true"` significa que a opção vira verdadeira quando aparece no comando.
- `if __name__ == "__main__":` chama `main()` apenas ao executar o arquivo diretamente. Importar suas funções nos testes não dispara treinamento.
- List comprehensions, como `[item["texto"] for item in mensagens]`, são uma forma curta de construir a lista que antes usava `append` em um laço.
- As funções retornam valores; suas variáveis internas não viram variáveis globais automaticamente.
- `None` representa a ausência de classificação ou de avaliação final no relatório.
- O hash identifica os bytes da base; não é uma métrica do modelo.

## Como falar do projeto

Uma descrição possível, depois de estudar e reproduzir:

> Desenvolvi um classificador de SMS em inglês com Python e scikit-learn durante meus estudos de IA. Validei os dados, tratei duplicatas e comparei Naive Bayes e regressão logística com TF-IDF. Separei treino, validação e teste e usei F1 de spam para escolher o modelo. Na validação, a regressão logística obteve F1 de 85,09%. Também documentei limitações e organizei testes e instruções para reprodução. O desenvolvimento contou com assistência de IA.

Não acrescente um resultado de teste sem executá-lo. Não descreva o projeto como treinamento de uma IA generativa, criação do algoritmo de regressão logística ou sistema já implantado em produção.

Para uma apresentação de cinco minutos: problema e dados → representação → separação e prevenção de vazamento → comparação e erros → demonstração → limitações. Pratique explicar cada etapa sem ler o código inteiro.

## Próximos estudos

Persistência do modelo e vetorizador juntos; uso de `Pipeline`; análise qualitativa de erros da validação; validação cruzada no desenvolvimento; comparação de representações; avaliação temporal ou por campanhas; interface simples. São possibilidades futuras, não funcionalidades já implementadas.

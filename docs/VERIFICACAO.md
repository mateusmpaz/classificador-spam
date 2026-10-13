# Verificação do pacote

Verificação realizada em 13/10/2026 durante a preparação do repositório, com assistência de IA. Os resultados abaixo foram efetivamente reproduzidos; não são uma nova medição no teste reservado.

## Ambiente da reprodução

- Windows, Python 3.12.14.
- scikit-learn 1.9.1.
- NumPy 2.5.3 e SciPy 1.18.1.

O estudante executou o experimento original com Python 3.14.7 e scikit-learn 1.9.1. A reprodução do pacote utilizou um ambiente já disponível, sem instalar novas dependências nesta preparação. O arquivo `requirements.txt` fixa a dependência direta; ele não é um congelamento completo de todas as dependências transitivas.

## Verificações executadas

1. **Nove testes automatizados passaram**, usando dados sintéticos.
2. O programa completo executou sobre a base incluída, com `--sem-interacao --relatorio docs/validacao_reproduzida.json`.
3. A leitura encontrou 5.574 registros; a limpeza manteve 5.171, com 653 spams e 4.518 hams.
4. A divisão reproduziu 3.102 / 1.034 / 1.035 registros em treino / validação / teste.
5. As matrizes de validação reproduziram `[[904, 0], [50, 80]]` para Naive Bayes e `[[903, 1], [33, 97]]` para regressão logística.
6. O vocabulário teve 6.516 termos na comparação e 7.669 no treinamento final.

O [relatório JSON](validacao_reproduzida.json) contém métricas sem arredondamento para exibição, tamanhos, versão do scikit-learn e hash da base. Seu campo `teste` é `null`, pois o teste real não foi avaliado nesta verificação.

No ambiente restrito de preparação, foi necessário apontar a pasta temporária dos testes para uma pasta gravável. Essa configuração não altera o código nem os resultados dos testes.

## O que ainda deve ser medido

A avaliação estatística final do modelo sobre o teste reservado. Ela está implementada, mas não há um resultado de teste publicado neste pacote. Use o comando documentado no README e preserve a distinção entre validação e teste ao apresentar o resultado.

Os testes automatizados verificam comportamentos do software; não certificam a qualidade do modelo em mensagens reais de outros contextos.

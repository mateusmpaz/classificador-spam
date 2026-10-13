# Classificador de spam com Python e aprendizado de máquina

Projeto de estudo de um estudante de Inteligência Artificial da UFCAT: classificar SMS em inglês como **ham** (mensagem legítima) ou **spam**, comparando Naive Bayes e regressão logística com representação TF-IDF.

O programa lê uma base pública, valida os registros, remove repetições exatas, separa os dados, compara modelos e permite digitar mensagens no terminal. O foco é aprender a construir e avaliar um experimento reproduzível de aprendizado supervisionado.

## Resultado principal

Na **validação**, a regressão logística obteve **85,09% de F1 para spam**, contra **76,19%** do Naive Bayes. Ela foi escolhida antes da avaliação do teste reservado.

| Modelo | Acurácia | Precisão de spam | Recall de spam | F1 de spam |
|---|---:|---:|---:|---:|
| Referência: prever sempre ham | 87,43% | 0,00%* | 0,00% | 0,00% |
| Multinomial Naive Bayes | 95,16% | 100,00% | 61,54% | 76,19% |
| Regressão logística | 96,71% | 98,98% | 74,62% | 85,09% |

*A referência não prevê spam: sua precisão para essa classe é indefinida e o código a representa como zero usando `zero_division=0`.*

Esses resultados pertencem às **1.034 mensagens de validação**. A avaliação das 1.035 mensagens de teste está **pendente neste relatório**. O programa já oferece um comando separado para executá-la. Nenhum resultado de validação é apresentado como resultado de teste.

Na regressão logística, a matriz de validação foi:

| Rótulo real / previsão | Ham | Spam |
|---|---:|---:|
| Ham | 903 | 1 |
| Spam | 33 | 97 |

O modelo identificou 97 dos 130 spams, deixou passar 33 e marcou uma mensagem legítima como spam. Uma acurácia alta, sozinha, não revelaria esses erros.

## Como executar no VS Code — Windows

1. Extraia ou clone o projeto e abra **a pasta que contém este README** no VS Code.
2. Abra **Terminal → Novo Terminal**.
3. Crie um ambiente virtual:

```powershell
python -m venv .venv
```

4. Instale a dependência do projeto. **Este comando baixa o scikit-learn e suas dependências** para o ambiente virtual:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

5. Execute:

```powershell
.\.venv\Scripts\python.exe classificador.py
```

O programa compara os modelos na validação, treina o modelo final com treino + validação e abre o campo `Mensagem:`. Digite um SMS em inglês. Use `/sair` para encerrar. Textos vazios ou sem termos reconhecidos não recebem classificação. As mensagens digitadas não são salvas nem usadas para atualizar o modelo.

Para usar o botão de execução: `Ctrl+Shift+P` → `Python: Select Interpreter` → escolha `.venv`. Depois abra `classificador.py` e execute o arquivo. Os comandos acima usam diretamente o Python do ambiente, sem exigir a ativação do PowerShell.

O experimento do estudante usou Python 3.14.7 e scikit-learn 1.9.1. A verificação do pacote está descrita em [docs/VERIFICACAO.md](docs/VERIFICACAO.md).

### Apenas treinar e salvar as métricas de validação

```powershell
.\.venv\Scripts\python.exe classificador.py --sem-interacao --relatorio resultados/validacao.json
```

### Avaliar o teste final

Execute depois de fixar a escolha do modelo e os critérios de avaliação:

```powershell
.\.venv\Scripts\python.exe classificador.py --avaliar-teste --sem-interacao --relatorio resultados/teste_final.json
```

O comando avalia também a referência sempre ham **no mesmo conjunto de teste**. Use o relatório para documentar o resultado final. Se você passar a ajustar o modelo com base nesses resultados, esse conjunto deixa de funcionar como avaliação independente dessas decisões.

Cada execução treina novamente. Esta versão não salva um modelo treinado em disco. O JSON salva métricas e metadados, não o modelo.

### Rodar os testes automatizados

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes usam exemplos sintéticos e verificam dados inválidos, conflitos, duplicatas, separação dos conjuntos, métricas e inferência sem termos reconhecidos. Eles não substituem a avaliação estatística do classificador.

## Dados e desenho do experimento

A base **SMS Spam Collection**, de Almeida e Hidalgo, está incluída para permitir execução local. Sua origem, licença e integridade estão documentadas em [dados/README.md](dados/README.md). Os dados são de terceiros; este projeto não os coletou nem os rotulou.

| Etapa | Mensagens |
|---|---:|
| Base original | 5.574 |
| Repetições exatas removidas | 403 |
| Base limpa | 5.171 |
| Ham na base limpa | 4.518 |
| Spam na base limpa | 653 |
| Treino | 3.102 |
| Validação | 1.034 |
| Teste reservado | 1.035 |

- Divisões estratificadas com `random_state=42`: aproximadamente 60% / 20% / 20%.
- Na comparação, o TF-IDF aprende vocabulário e pesos somente no treino: 6.516 termos.
- Na validação, usa-se apenas `transform`, sem reaprender a representação.
- Com a regressão logística escolhida, o TF-IDF e o modelo são treinados novamente nas 4.136 mensagens de desenvolvimento: 7.669 termos.
- O teste fica fora do ajuste dos modelos e do vocabulário.
- A comparação usa os padrões de `TfidfVectorizer()` e `MultinomialNB()`, e `LogisticRegression(max_iter=1000, random_state=42)`. Não houve busca de hiperparâmetros.

## Organização

```text
classificador-spam/
├── classificador.py           # Leitura, treinamento, avaliação e terminal
├── requirements.txt          # Dependência direta fixada
├── .gitignore                # Arquivos locais que não entram no Git
├── README.md
├── PUBLICAR_NO_GITHUB.md
├── dados/
│   ├── SMSSpamCollection      # Base pública original
│   ├── README_original.txt   # Documentação distribuída com a base
│   └── README.md             # Origem, licença e hash
├── docs/
│   ├── GUIA_DE_ESTUDO.md
│   ├── VERIFICACAO.md
│   └── validacao_reproduzida.json
└── tests/
    └── test_classificador.py
```

## Limitações

- Base de SMS em inglês: não foi demonstrada qualidade em português, e-mails, WhatsApp ou mensagens atuais de outros contextos.
- Spam não é sinônimo de fraude. Uma previsão ham não garante segurança.
- A remoção trata textos exatamente iguais. Mensagens quase iguais ou da mesma campanha ainda podem aparecer em conjuntos diferentes.
- Uma única divisão aleatória não mede toda a variação possível. Não há validação cruzada, intervalo de confiança nem avaliação temporal.
- O programa não possui integração com serviços de mensagens, interface web, aprendizado contínuo ou implantação em produção.
- O TF-IDF usa termos individuais e não representa o contexto como um modelo de linguagem. Uma palavra presente não determina sozinha a classe.
- A seed ajuda a repetir a divisão; base, ordem dos registros e versões também influenciam a reprodução.

## Aprendizado e autoria

Projeto desenvolvido em um percurso de estudo com assistência de IA para explicações, depuração, organização e documentação. O trabalho utiliza algoritmos implementados pelo scikit-learn. A contribuição do projeto é preparar os dados, montar o experimento, comparar resultados e construir uma aplicação local de classificação.

Consulte o [guia de estudo e apresentação](docs/GUIA_DE_ESTUDO.md) para entender as decisões e praticar uma explicação do código.

## Referências

- Almeida, T. & Hidalgo, J. (2011). [SMS Spam Collection — UCI](https://doi.org/10.24432/C5CC84).
- [TfidfVectorizer — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
- [Naive Bayes — scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html).
- [Regressão logística — scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
- [Cuidados com vazamento de dados — scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).

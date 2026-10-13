"""Classificador educacional de SMS em inglês. Execute este arquivo no VS Code."""

import argparse
import hashlib
import json
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def carregar_mensagens(caminho):
    """Lê o arquivo rótulo TAB texto e rejeita registros inválidos."""
    mensagens = []
    with Path(caminho).open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            linha = linha.rstrip("\r\n")
            if not linha.strip():
                continue
            partes = linha.split("\t", 1)
            if len(partes) != 2:
                raise ValueError(f"Linha {numero}: faltou a tabulação entre rótulo e texto.")
            rotulo, texto = partes
            if rotulo not in ("ham", "spam"):
                raise ValueError(f"Linha {numero}: rótulo desconhecido: {rotulo!r}.")
            if not texto.strip():
                raise ValueError(f"Linha {numero}: mensagem sem texto.")
            mensagens.append({"rotulo": rotulo, "texto": texto})
    if not mensagens:
        raise ValueError("A base não contém mensagens.")
    return mensagens


def contar_rotulos(mensagens):
    contagem = {"ham": 0, "spam": 0}
    for item in mensagens:
        contagem[item["rotulo"]] += 1
    return contagem


def remover_duplicatas(mensagens):
    """Mantém a primeira ocorrência; interrompe se houver rótulos conflitantes."""
    rotulo_por_texto = {}
    limpas = []
    for item in mensagens:
        texto = item["texto"]
        rotulo = item["rotulo"]
        if texto not in rotulo_por_texto:
            rotulo_por_texto[texto] = rotulo
            limpas.append(item)
        elif rotulo_por_texto[texto] != rotulo:
            raise ValueError("Há textos iguais com rótulos diferentes. Investigue a base.")
    return limpas


def dividir_dados(mensagens):
    textos = [item["texto"] for item in mensagens]
    rotulos = [item["rotulo"] for item in mensagens]
    textos_dev, textos_teste, rotulos_dev, rotulos_teste = train_test_split(
        textos, rotulos, test_size=0.20, random_state=42, stratify=rotulos,
    )
    textos_treino, textos_val, rotulos_treino, rotulos_val = train_test_split(
        textos_dev, rotulos_dev, test_size=0.25,
        random_state=42, stratify=rotulos_dev,
    )
    return {
        "treino": (textos_treino, rotulos_treino),
        "validacao": (textos_val, rotulos_val),
        "teste": (textos_teste, rotulos_teste),
        "desenvolvimento": (textos_dev, rotulos_dev),
    }


def calcular_metricas(rotulos_reais, previsoes):
    return {
        "matriz_confusao": confusion_matrix(
            rotulos_reais, previsoes, labels=["ham", "spam"],
        ).tolist(),
        "acuracia": float(accuracy_score(rotulos_reais, previsoes)),
        "precisao_spam": float(precision_score(
            rotulos_reais, previsoes, pos_label="spam", zero_division=0,
        )),
        "recall_spam": float(recall_score(
            rotulos_reais, previsoes, pos_label="spam", zero_division=0,
        )),
        "f1_spam": float(f1_score(
            rotulos_reais, previsoes, pos_label="spam", zero_division=0,
        )),
    }


def mostrar_metricas(titulo, metricas):
    print(f"\n{titulo}")
    print("Matriz: linhas = reais; colunas = previstas; ordem = ham, spam.")
    for linha in metricas["matriz_confusao"]:
        print(linha)
    for chave in ("acuracia", "precisao_spam", "recall_spam", "f1_spam"):
        print(f"{chave}: {metricas[chave]:.2%}")


def classificar_texto(texto, vetorizador, modelo):
    """Retorna ham/spam, ou None se o texto não oferece termos reconhecidos."""
    if not texto.strip():
        return None
    representacao = vetorizador.transform([texto])
    if representacao.nnz == 0:
        return None
    return str(modelo.predict(representacao)[0])


def modo_interativo(vetorizador, modelo):
    print("\nDigite um SMS em inglês, ou /sair para encerrar.")
    print("O modelo pode errar; HAM é uma previsão, não uma garantia de segurança.")
    while True:
        try:
            texto = input("\nMensagem: ")
        except (EOFError, KeyboardInterrupt):
            print("\nPrograma encerrado.")
            break
        if texto.strip().lower() == "/sair":
            break
        previsao = classificar_texto(texto, vetorizador, modelo)
        if previsao is None:
            print("Sem texto ou sem termos reconhecidos. Não há classificação.")
        else:
            print(f"Previsão do modelo: {previsao.upper()}")


def executar(caminho, avaliar_teste=False):
    """Compara na validação e treina o modelo final no desenvolvimento."""
    mensagens = carregar_mensagens(caminho)
    limpas = remover_duplicatas(mensagens)
    conjuntos = dividir_dados(limpas)
    print("Quantidade original:", len(mensagens))
    print("Contagem original:", contar_rotulos(mensagens))
    print("Repetições removidas:", len(mensagens) - len(limpas))
    print("Quantidade sem duplicatas:", len(limpas))
    print("Contagem após limpeza:", contar_rotulos(limpas))
    print("Textos com rótulos conflitantes: 0 (verificação concluída)")
    for nome, (textos, _) in conjuntos.items():
        print(f"Mensagens de {nome}: {len(textos)}")

    textos_treino, rotulos_treino = conjuntos["treino"]
    textos_val, rotulos_val = conjuntos["validacao"]
    vetorizador = TfidfVectorizer()
    # Somente o treino pode definir o vocabulário e os pesos IDF nesta etapa.
    X_treino = vetorizador.fit_transform(textos_treino)
    X_val = vetorizador.transform(textos_val)
    print("Formato do treino:", X_treino.shape)
    print("Formato da validação:", X_val.shape)

    modelos = {
        "naive_bayes": MultinomialNB(),
        "regressao_logistica": LogisticRegression(max_iter=1000, random_state=42),
    }
    validacao = {
        "sempre_ham": calcular_metricas(rotulos_val, ["ham"] * len(rotulos_val)),
    }
    mostrar_metricas("REFERÊNCIA SEMPRE HAM — VALIDAÇÃO", validacao["sempre_ham"])
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, rotulos_treino)
        validacao[nome] = calcular_metricas(rotulos_val, modelo.predict(X_val))
        mostrar_metricas(f"{nome.upper()} — VALIDAÇÃO", validacao[nome])

    # A escolha foi feita no experimento original usando o F1 de validação.
    # Ela permanece fixa: não escolhemos nem ajustamos o modelo pelo teste.
    textos_dev, rotulos_dev = conjuntos["desenvolvimento"]
    vetorizador_final = TfidfVectorizer()
    X_dev = vetorizador_final.fit_transform(textos_dev)
    modelo_final = LogisticRegression(max_iter=1000, random_state=42)
    modelo_final.fit(X_dev, rotulos_dev)
    print("\nTreinamento final concluído:", X_dev.shape)

    relatorio = {
        "scikit_learn": sklearn.__version__,
        "sha256_base": hashlib.sha256(Path(caminho).read_bytes()).hexdigest(),
        "seed": 42,
        "quantidade_original": len(mensagens),
        "quantidade_sem_duplicatas": len(limpas),
        "contagem_limpa": contar_rotulos(limpas),
        "tamanhos": {nome: len(par[0]) for nome, par in conjuntos.items()},
        "vocabulario_treino": X_treino.shape[1],
        "vocabulario_final": X_dev.shape[1],
        "modelo_final": "regressao_logistica",
        "criterio_escolha": "F1 de spam na validação do experimento original",
        "validacao": validacao,
        "teste": None,
    }
    if avaliar_teste:
        textos_teste, rotulos_teste = conjuntos["teste"]
        X_teste = vetorizador_final.transform(textos_teste)
        relatorio["teste"] = {
            "regressao_logistica": calcular_metricas(
                rotulos_teste, modelo_final.predict(X_teste),
            ),
            "sempre_ham": calcular_metricas(
                rotulos_teste, ["ham"] * len(rotulos_teste),
            ),
        }
        for nome, metricas in relatorio["teste"].items():
            mostrar_metricas(f"{nome.upper()} — TESTE FINAL", metricas)
    else:
        print("Teste final não avaliado nesta execução. Use --avaliar-teste para avaliar.")
    return vetorizador_final, modelo_final, relatorio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dados", type=Path,
        default=Path(__file__).resolve().parent / "dados" / "SMSSpamCollection",
        help="Arquivo no formato rótulo TAB texto.",
    )
    parser.add_argument("--avaliar-teste", action="store_true", help="Avalia o teste reservado.")
    parser.add_argument("--sem-interacao", action="store_true", help="Encerra após treinar/avaliar.")
    parser.add_argument("--relatorio", type=Path, help="Salva métricas em JSON no caminho indicado.")
    args = parser.parse_args()
    try:
        vetorizador, modelo, relatorio = executar(args.dados, args.avaliar_teste)
        if args.relatorio:
            args.relatorio.parent.mkdir(parents=True, exist_ok=True)
            args.relatorio.write_text(
                json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            print("Relatório salvo em:", args.relatorio)
    except (OSError, ValueError) as erro:
        parser.exit(1, f"Erro: {erro}\n")
    if not args.sem_interacao:
        modo_interativo(vetorizador, modelo)


if __name__ == "__main__":
    main()

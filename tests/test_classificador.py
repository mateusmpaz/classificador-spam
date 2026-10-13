"""Testes com dados sintéticos: não usam o conjunto reservado de avaliação."""

import tempfile
import unittest
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from classificador import (
    calcular_metricas, carregar_mensagens, classificar_texto,
    dividir_dados, remover_duplicatas,
)


class TestDados(unittest.TestCase):
    def carregar_conteudo(self, conteudo):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / "mensagens.txt"
            arquivo.write_text(conteudo, encoding="utf-8")
            return carregar_mensagens(arquivo)

    def test_preserva_tabulacao_no_texto_e_ignora_linha_vazia(self):
        mensagens = self.carregar_conteudo("\nham\thello\tfriend\n")
        self.assertEqual(mensagens, [{"rotulo": "ham", "texto": "hello\tfriend"}])

    def test_rejeita_base_vazia(self):
        with self.assertRaisesRegex(ValueError, "não contém"):
            self.carregar_conteudo("\n   \n")

    def test_rejeita_registros_invalidos(self):
        for conteudo in ("ham sem tab", "outro\thello", "spam\t   "):
            with self.subTest(conteudo=conteudo):
                with self.assertRaisesRegex(ValueError, "Linha 1"):
                    self.carregar_conteudo(conteudo)

    def test_remove_repeticoes_sem_mudar_a_ordem(self):
        a = {"texto": "hello", "rotulo": "ham"}
        b = {"texto": "prize", "rotulo": "spam"}
        self.assertEqual(remover_duplicatas([a, b, a]), [a, b])

    def test_conflito_nao_e_resolvido_silenciosamente(self):
        with self.assertRaisesRegex(ValueError, "rótulos diferentes"):
            remover_duplicatas([
                {"texto": "same", "rotulo": "ham"},
                {"texto": "same", "rotulo": "spam"},
            ])

    def test_particoes_disjuntas_reproduziveis_e_alinhadas(self):
        mensagens = [
            {"texto": f"mensagem {i}", "rotulo": "spam" if i % 5 == 0 else "ham"}
            for i in range(100)
        ]
        partes = dividir_dados(mensagens)
        self.assertEqual(partes, dividir_dados(mensagens))
        treino, val, teste = [set(partes[n][0]) for n in ("treino", "validacao", "teste")]
        self.assertFalse(treino & val or treino & teste or val & teste)
        self.assertEqual(len(treino | val | teste), 100)
        self.assertEqual(treino | val, set(partes["desenvolvimento"][0]))
        original = {m["texto"]: m["rotulo"] for m in mensagens}
        for textos, rotulos in partes.values():
            self.assertEqual(rotulos, [original[t] for t in textos])
            self.assertEqual(set(rotulos), {"ham", "spam"})


class TestAvaliacao(unittest.TestCase):
    def test_baseline_nao_detecta_spam(self):
        metricas = calcular_metricas(["ham", "spam", "ham"], ["ham"] * 3)
        self.assertEqual(metricas["matriz_confusao"], [[2, 0], [1, 0]])
        self.assertEqual(metricas["recall_spam"], 0)
        self.assertEqual(metricas["f1_spam"], 0)

    def test_ordem_da_matriz_e_metrica_da_classe_spam(self):
        metricas = calcular_metricas(
            ["ham", "ham", "spam", "spam", "spam"],
            ["ham", "spam", "ham", "spam", "spam"],
        )
        self.assertEqual(metricas["matriz_confusao"], [[1, 1], [1, 2]])
        self.assertAlmostEqual(metricas["f1_spam"], 2 / 3)

    def test_inferencia_sem_termos_nao_inventa_rotulo(self):
        vetorizador = TfidfVectorizer()
        X = vetorizador.fit_transform(["hello friend", "free prize"])
        modelo = MultinomialNB().fit(X, ["ham", "spam"])
        self.assertIsNone(classificar_texto("", vetorizador, modelo))
        self.assertIsNone(classificar_texto("zzzzqqqq", vetorizador, modelo))
        self.assertEqual(classificar_texto("free prize", vetorizador, modelo), "spam")


if __name__ == "__main__":
    unittest.main()

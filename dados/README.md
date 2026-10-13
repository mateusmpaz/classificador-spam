# Base pública utilizada

**SMS Spam Collection** — Almeida, T. & Hidalgo, J. (2011).

- Página oficial: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- DOI: https://doi.org/10.24432/C5CC84
- Licença informada pela UCI: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- Citação: Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

O arquivo `SMSSpamCollection` incluído foi copiado da base pública usada no experimento, sem alteração. O `README_original.txt` preserva a documentação que acompanhava essa cópia. A limpeza de duplicatas acontece em memória, ao executar o programa, e não modifica o arquivo original.

Cada linha contém um rótulo (`ham` ou `spam`), uma tabulação e a mensagem. Não há cabeçalho. A cópia utilizada contém 5.574 registros. A fonte e os rótulos são de terceiros, não do autor deste projeto.

SHA-256 do arquivo incluído:

```text
7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d
```

Para conferir no PowerShell, a partir da raiz do projeto:

```powershell
Get-FileHash .\dados\SMSSpamCollection -Algorithm SHA256
```

Se o arquivo estiver faltando, baixe a base na página oficial, extraia o ZIP e coloque `SMSSpamCollection` nesta pasta, sem acrescentar a extensão `.txt`. O programa não faz downloads automáticos.

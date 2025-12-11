# Regressão do consumo de energia em ETE de Melbourne

Este repositório contém o projeto do Homework 2 da disciplina ICA, cujo objetivo
é comparar modelos de regressão para prever o consumo diário de energia elétrica
(`total_grid`) de uma estação de tratamento de esgoto (ETE) em Melbourne
utilizando variáveis operacionais e meteorológicas diárias (2014–2019).

O relatório final (formato IEEE) está em:

- `report/hw2_regression_Melbourne.pdf`

---

## Estrutura do repositório
.
├─ report/
│ ├─ hw2_regression_Melbourne.tex # fonte LaTeX do relatório
│ └─ hw2_regression_Melbourne.pdf # relatório final em PDF
├─ code/
│ ├─ hw2_preprocess_melbourne.py # limpeza do CSV bruto e pré-processamento
│ └─ hw2_models_melbourne.py # treino dos modelos e geração dos resultados
├─ data/
│ └─ Data-Melbourne_F_clean.csv # dataset diário limpo (se permitido)
├─ outputs_hw2/
│ ├─ train_preprocessed.csv # treino após log(PP) e padronização
│ ├─ test_preprocessed.csv # teste após log(PP) e padronização
│ └─ hw2_models_summary.csv # resumo de RMSE e R² no teste
└─ README.md

---

## Dependências

As principais dependências em Python são:

- `python` (>= 3.9)
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib` (apenas para geração de gráficos opcionais)

Um arquivo `requirements.txt` pode ser criado com:

numpy
pandas
scikit-learn
matplotlib
---

## Como executar o código

### 1. Criar ambiente virtual (opcional, mas recomendado)
python -m venv .venv

Linux/macOS:
source .venv/bin/activate

Windows:
.\venv\Scripts\activate

### 2. Instalar dependências
pip install -r requirements.txt

### 3. Preparar dados

Colocar o arquivo original `Data-Melbourne_F.csv` em algum diretório
(e ajustar o caminho em `code/hw2_preprocess_melbourne.py` caso necessário).

Em seguida, executar:

python code/hw2_preprocess_melbourne.py

Este script:

- lê o arquivo bruto `Data-Melbourne_F.csv`;
- remove duplicatas, valores ausentes, consumos não positivos e outliers extremos de `total_grid`;
- divide os dados em treino (75%) e teste (25%);
- aplica `log(1 + x)` em `PP` e padronização z-score;
- salva:
  - `data/Data-Melbourne_F_clean.csv`;
  - `outputs_hw2/train_preprocessed.csv`;
  - `outputs_hw2/test_preprocessed.csv`.

### 4. Rodar os modelos

python code/hw2_models_melbourne.py
Este script:

- carrega `train_preprocessed.csv` e `test_preprocessed.csv`;
- ajusta os modelos OLS, ridge, PCR, PLS e MLP;
- realiza validação cruzada 5-fold para escolher:
  - `lambda` (ridge),
  - número de componentes (PCR/PLS);
- calcula RMSE e R² no conjunto de teste;
- imprime os resultados no terminal;
- salva `outputs_hw2/hw2_models_summary.csv`, usado pelo LaTeX
  para gerar a tabela de resultados.

## Contribuições

Trabalho individual. Todas as etapas do projeto (limpeza de dados, implementação
dos modelos, análise dos resultados e escrita do relatório) foram realizadas por:

- **Luiz Augusto Gomes da Silva de Jesus**


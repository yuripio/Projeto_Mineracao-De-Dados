# 📘 README - Análise Preditiva de Tipos de Acidentes Utilizando Random Forest

Jhully Vitória Nunes Leite
João Vitor Gonçalves
Lucas Daniel da Cunha Moura
Victor Hugo Nunes
Yuri Pio

## ✅ Requisitos para a execução em Linux

Sistema operacional:
- Linux Fedora 42 **ou**
- Ubuntu 24.04

Dependências Python:
- Python 3.10 ou superior
- pandas
- matplotlib
- seaborn
- scikit-learn

## 🔧 Passo a Passo para Instalação e Execução

### 1. Instale o Python (caso ainda não esteja instalado)

No **Ubuntu 24.04**:
```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

No **Fedora 42**:
```bash
sudo dnf install python3 python3-pip -y
```

### 2. Crie e ative um ambiente virtual (recomendado)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as bibliotecas necessárias

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## ▶️ Como Executar os Arquivos

Certifique-se de que o arquivo `DataSetNormalizado.csv` está no mesmo diretório que os arquivos `.py`.

### Executar o modelo Random Forest:
```bash
python3 RandomForest.py
```

### Executar o modelo KNN:
```bash
python3 KNN.py
```

Ambos os scripts:
- Carregam os dados do arquivo `DataSetNormalizado.csv`
- Aplicam pré-processamento e codificação dos rótulos
- Treinam e testam o modelo (Random Forest ou KNN)
- Mostram a acurácia, relatório de classificação e matriz de confusão
- No caso do Random Forest, também exibe um gráfico da importância das features e visualização de árvore
#  QuintikusOpen

**Transformer deterministo – Dlm class 1‑2**  
**Quintikus Proton-Flow TDLM v117.0 [MIT]**

![Modelo Dual Loop Memory F](https://github.com/beta-test-Ronan/QuintikusOpen/blob/main/model-Dlm-f.png?raw=true)

---

## 📌 Visão Geral

O sistema implementa uma arquitetura de **Dupla Inteligência Artificial** com suporte a blockchain de conhecimento:

- **QuintikusSovereignCore** – IA Analítica, baseada em camadas de visão e massa geradas a partir de um texto fonte.
- **QuintikusAGI** – IA Emocional, com estados térmicos internos, memória associativa (DLM – *Dual Loop Memory*) e personalidade dinâmica.

Ambas as IAs partilham o mesmo parágrafo de treino extraído do núcleo analítico, mas operam com paradigmas distintos:  
uma responde de forma **lógica** e a outra de forma **emotiva**.

---

## 📦 Dependências

| Biblioteca          | Utilização no código atual                            |
|---------------------|-------------------------------------------------------|
| `numpy`             | Processamento linear                                  |
| `hashlib`           | Geração de hashes de integridade e identificadores    |
| `time`              | Simulação de processamento e métricas                 |
| `pickle`            | Serialização da blockchain                            |
| `os`                | Manipulação de arquivos                               |
| `re`                | Processamento de texto (expressões regulares)         |
| `random`            | Escolha aleatória de frases dos arquétipos            |
| `sys`,`unicodedata` | Config system                                         |

## ⚙️ Estrutura de Classes

### 1. SovereignBlockchain
Responsável pela persistência e integridade do conhecimento gerado.

| Método                          | Descrição                                                                                  |
|---------------------------------|--------------------------------------------------------------------------------------------|
| `__init__(name)`                | Define o caminho do ficheiro de cache (`blockchain_{name}.cache`).                         |
| `selar_memoria(knowledge_bundle)` | Serializa o *bundle* com `pickle` e retorna o hash SHA‑256 (8 caracteres) como assinatura. |
| `carregar_ponteiro()`           | Carrega o *bundle* do disco, se o ficheiro existir.                                        |

### 2. QuintikusSovereignCore (Núcleo Analítico)
Armazena o conhecimento em três camadas:

- `layer1_vision` – visões sintéticas com ponteiros, entropia e janelas de contexto.
- `layer2_mass` – factos brutos originais.
- `word_rarity` – raridade de palavras baseada na frequência.

#### 🧠 Métodos Principais

| Método                                    | Descrição                                                                                           |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `amadurecer_nexo(raw_text)`               | Processa o texto bruto, divide‑o em fragmentos, calcula entropia e povoa `layer1_vision` e `layer2_mass`. Gera *bundle* para a blockchain. |
| `falar_soberano(pergunta, cache)`         | Gera respostas comparando a entropia da pergunta com a **Camada 1**. Usa os arquétipos para a saída textual. |
| `carregar_fundamentos(mc_f, mm_f)`        | Carrega os ficheiros de arquétipos e preenche as definições de personalidade.                      |
| `exportar_banco_normalizado()`            | Converte os factos da `layer2_mass` num parágrafo único e contínuo, livre de duplicados.           |
| `texto()`                                 | Gera um relatório do estado do núcleo para depuração (*debug*).                                    |

> 📖 **Arquétipos de Personalidade**  
> Os ficheiros `mc.txt` e `mm.txt` definem o comportamento linguístico. São carregados no formato 
> `<intro>`, `<ponte>`, `<concl>` para popular o atributo `self.arquetipos`.

### 3. QuintikusAGI (Núcleo Emocional)
Implementa um **estado interno de temperatura emocional** (três variáveis), um dicionário de palavras com força e raridade, e uma memória em cadeia (**DLM**) que liga épocas de contexto.

| Método               | Descrição                                                                                                              |
|----------------------|------------------------------------------------------------------------------------------------------------------------|
| `__init__(_t)`       | Inicializa estados térmicos, dicionários e blocos filosóficos em hexadecimal.                                         |
| `inicializar(_txt)`  | Analisa o parágrafo de entrada: estatísticas de palavras, criação de épocas (grupos de 5 frases) e construção da DLM. |
| `_upd_thermal(_q)`   | Atualiza os estados de estresse e harmonia com base nas palavras‑chave da pergunta (tabela de valências).             |
| `falar(_qi)`         | Gera uma resposta usando a melhor época que intersecta a pergunta, aplica a DLM e escolhe frases de abertura/fecho conforme o estado térmico. |

---

## 🔁 Fluxo de Execução Principal

1. **Inicialização**  
   - Instancia `QuintikusSovereignCore` e carrega arquétipos (`mc.txt`, `mm.txt`).  
   - Lê `texto.txt` (ou usa texto padrão).  
   - Se não existir blockchain em cache, processa o texto via `amadurecer_nexo` e sela.

2. **Preparação do Parágrafo Único**  
   - Carrega a memória ativa da blockchain.  
   - Gera um parágrafo contínuo com `exportar_banco_normalizado()`.  
   - Exibe os primeiros 500 caracteres.

3. **Inicialização da AGI Emocional**  
   - Cria uma instância de `QuintikusAGI` e chama `inicializar()` com o mesmo parágrafo usado pelo núcleo analítico.

4. **Loop Interativo**  
   - Pergunta ao utilizador (`RONAN:`).  
   - Obtém resposta **lógica** de `SovereignCore.falar_soberano()`.  
   - Obtém resposta **emocional** de `AGI.falar()`, passando o comando e o texto original.  
   - Exibe a resposta da AGI com efeito de digitação (`efeito_llm`).

---

## 📁 Arquivos Necessários

| Arquivo                       | Função                                                     |
|-------------------------------|------------------------------------------------------------|
| `texto.txt`                   | Texto fonte para treino do núcleo analítico.               |
| `mc.txt`                      | Arquétipos de personalidade *"mc"* (intro / ponte / concl).|
| `mm.txt`                      | Arquétipos de personalidade *"mm"*.                        |
| `blockchain_machado.cache`    | Cache da blockchain (gerado automaticamente).              |



## ✍️ Formato dos Arquétipos (`mc.txt`, `mm.txt`)
Os ficheiros devem conter marcadores como os seguintes:


<intro>
Frase de introdução 1
Frase de introdução 2
<ponte>
...
<concl>
...




## 💬 Exemplo de Uso


📄 PARÁGRAFO GERADO (mesmo texto para as duas IAs):

Saída típica durante a interação:
text

👤 RONAN: Qual o sentido do fluxo galvânico?

💡 [CARDUS MASTER FLOW | 0.15 μs | Quality: 100%]
LAYER-1 (VISÃO): Analisei que Localizado nexo no ponteiro 12345678.
LAYER-2 (MASSA): O fluxo galvânico inicializa o sistema sem base externa.
 | Pulse | [FLUXO] <-> [GALVÂNICO] | Densidade: 0.9234
-> fim. (Selo: Cardus-100)
[DLM-FLOW: 45.23μs | D:8/10 | T:0.2 | DLM-ACTIVE | SIGN: 25e0bb26]
No vácuo, o fluxo galvânico inicializa o sistema sem base externa. Além disso, ... Aguardando nexo.


## 📝 Notas Técnicas

    A IA analítica utiliza hash e entropia para associar perguntas a factos.

    A IA emocional usa uma rede de memória temporal (DLM) que liga épocas consecutivas.

    O estado térmico (self._st) influencia a escolha de frases e o tom da resposta.

    Todo o conhecimento da Camada 2 é compactado num parágrafo normalizado que alimenta a AGI, garantindo consistência total entre os dois núcleos.

## Model

   Agi = precessamento bruto e busca<br>
   Fast = velocidade de busca grande contexto 

Versão: Quintikus Proton-Flow TDLM v117.0
Autor: Ronan Basto
Licença: Livre para estudo e experimentação.

🧠 QuintikusOpen — Dual AI Architecture with Knowledge Blockchain

> Transformer · LLM · CHATBOT · Topology Geometric Processing
> Class Dlm 1–2 — Quintikus class [TDLM‑TGP‑SSML‑DLMC‑DSLM] — [MIT]


🌐 Translate this page

[English](https://translate.google.com/translate?hl=en&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) ·
[Español](https://translate.google.com/translate?hl=es&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) ·
[中文 (简)](https://translate.google.com/translate?hl=zh-CN&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) ·
[中文 (繁)](https://translate.google.com/translate?hl=zh-TW&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) ·
[Africâner](https://translate.google.com/translate?hl=af&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) ·
[Swahili](https://translate.google.com/translate?hl=sw&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) ·
[Zulu](https://translate.google.com/translate?hl=zu&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) ·
[Yorùbá](https://translate.google.com/translate?hl=yo&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/)


📌 Visão Geral

O sistema implementa uma arquitetura de Dupla Inteligência Artificial com suporte a blockchain de conhecimento:

| IA                           | Paradigma          | Descrição                                                                                   |
|------------------------------|--------------------|---------------------------------------------------------------------------------------------|
| QuintikusSovereignCore       | Analítico / Lógico | Baseado em camadas de visão e massa geradas a partir de um texto fonte.                     |
| QuintikusAGI                 | Emocional / Térmico| Estados térmicos internos, memória associativa (DLM – Dual Loop Memory) e personalidade dinâmica. |

Ambas partilham o mesmo parágrafo de treino extraído do núcleo analítico, mas operam com paradigmas distintos:
uma responde de forma lógica e a outra de forma emotiva.


[Imagem: Modelo Dual Loop Memory F]
https://github.com/beta-test-Ronan/QuintikusOpen/blob/main/model-Dlm-f.png?raw=true


📦 Dependências

| Biblioteca       | Utilização no código atual                         |
|------------------|----------------------------------------------------|
| numpy            | Processamento linear                               |
| hashlib          | Geração de hashes de integridade e identificadores |
| time             | Simulação de processamento e métricas              |
| pickle           | Serialização da blockchain                         |
| os               | Manipulação de arquivos                            |
| re               | Processamento de texto (expressões regulares)      |
| random           | Escolha aleatória de frases dos arquétipos         |
| sys, unicodedata | Configuração do sistema                            |


🧱 Estrutura de Classes

Quintikus-Agi_base → [SovereignBlockchain Cache]
                        │
                [ Parágrafo Único Normalizado ]
                        │
          ┌─────────────┴─────────────┐
          ▼                             ▼
[ QuintikusSovereignCore ]          [ QuintikusAGI ]
├─ Paradigma: Lógico/Analítico    ├─ Paradigma: Emocional/Térmico
├─ Camada 1: Visão (Entropia)     ├─ Estrutura: Dual Loop Memory (DLM)
└─ Camada 2: Massa (Fatos brutos) └─ Dinâmica: Estados Térmicos Mutáveis

1. SovereignBlockchain

Responsável pela persistência e integridade do conhecimento gerado.

| Método                      | Descrição                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------|
| __init__(name)              | Define o caminho do ficheiro de cache (blockchain_{name}.cache).                           |
| selar_memoria(knowledge_bundle) | Serializa o bundle com pickle e retorna o hash SHA‑256 (8 caracteres) como assinatura. |
| carregar_ponteiro()         | Carrega o bundle do disco, se o ficheiro existir.                                          |


2. QuintikusSovereignCore — Núcleo Analítico

Armazena o conhecimento em três camadas:
  - layer1_vision – visões sintéticas com ponteiros, entropia e janelas de contexto.
  - layer2_mass   – factos brutos originais.
  - word_rarity   – raridade de palavras baseada na frequência.

🧠 Métodos Principais

| Método                         | Descrição                                                                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| amadurecer_nexo(raw_text)      | Processa o texto bruto, divide‑o em fragmentos, calcula entropia e povoa layer1_vision e layer2_mass. Gera bundle para a blockchain.|
| falar_soberano(pergunta, cache)| Gera respostas comparando a entropia da pergunta com a Camada 1. Usa os arquétipos para a saída textual.                           |
| carregar_fundamentos(mc_f, mm_f)| Carrega os ficheiros de arquétipos e preenche as definições de personalidade.                                                      |
| exportar_banco_normalizado()   | Converte os factos da layer2_mass num parágrafo único e contínuo, livre de duplicados.                                              |
| texto()                        | Gera um relatório do estado do núcleo para depuração (debug).                                                                       |

📖 Arquétipos de Personalidade
Os ficheiros mc.txt e mm.txt definem o comportamento linguístico. São carregados no formato <intro>, <ponte>, <concl> para popular o atributo self.arquetipos.


3. QuintikusAGI — Núcleo Emocional

Implementa um estado interno de temperatura emocional (três variáveis), um dicionário de palavras com força e raridade, e uma memória em cadeia (DLM) que liga épocas de contexto.

| Método           | Descrição                                                                                                                               |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| __init__(_t)     | Inicializa estados térmicos, dicionários e blocos filosóficos em hexadecimal.                                                           |
| inicializar(_txt)| Analisa o parágrafo de entrada: estatísticas de palavras, criação de épocas (grupos de 5 frases) e construção da DLM.                   |
| _upd_thermal(_q) | Atualiza os estados de estresse e harmonia com base nas palavras‑chave da pergunta (tabela de valências).                               |
| falar(_qi)       | Gera uma resposta usando a melhor época que intersecta a pergunta, aplica a DLM e escolhe frases de abertura/fecho conforme o estado térmico. |


🔁 Fluxo de Execução Principal

1. Inicialização
   - Instancia QuintikusSovereignCore e carrega arquétipos (mc.txt, mm.txt).
   - Lê texto.txt (ou usa texto padrão).
   - Se não existir blockchain em cache, processa o texto via amadurecer_nexo e sela.

2. Preparação do Parágrafo Único
   - Carrega a memória ativa da blockchain.
   - Gera um parágrafo contínuo com exportar_banco_normalizado().
   - Exibe os primeiros 500 caracteres.

3. Inicialização da AGI Emocional
   - Cria uma instância de QuintikusAGI e chama inicializar() com o mesmo parágrafo usado pelo núcleo analítico.

4. Loop Interativo
   - Pergunta ao utilizador (RONAN:).
   - Obtém resposta lógica de SovereignCore.falar_soberano().
   - Obtém resposta emocional de AGI.falar(), passando o comando e o texto original.
   - Exibe a resposta da AGI com efeito de digitação (efeito_llm).


📁 Arquivos Necessários

| Arquivo                   | Função                                                                 |
|---------------------------|------------------------------------------------------------------------|
| texto.txt                 | Texto fonte para treino do núcleo analítico.                           |
| mc.txt                    | Arquétipos de personalidade "mc" (intro / ponte / concl).              |
| mm.txt                    | Arquétipos de personalidade "mm".                                      |
| blockchain_machado.cache  | Cache da blockchain (gerado automaticamente).                          |

✍️ Formato dos Arquétipos (mc.txt, mm.txt)

Os ficheiros devem conter marcadores como:
-------------------------------------------------
<intro>
Frase de introdução 1
Frase de introdução 2
<ponte>
...
<concl>
...
-------------------------------------------------


💬 Exemplo de Uso

📄 PARÁGRAFO GERADO (mesmo texto para as duas IAs):

Saída típica durante a interação:
-------------------------------------------------
👤 RONAN: Qual o sentido do fluxo galvânico?

💡 [CARDUS MASTER FLOW | 0.15 μs | Quality: 100%]
LAYER-1 (VISÃO): Analisei que Localizado nexo no ponteiro 12345678.
LAYER-2 (MASSA): O fluxo galvânico inicializa o sistema sem base externa.
 | Pulse | [FLUXO] <-> [GALVÂNICO] | Densidade: 0.9234
-> fim. (Selo: Cardus-100)
[DLM-FLOW: 45.23μs | D:8/10 | T:0.2 | DLM-ACTIVE | SIGN: 25e0bb26]
No vácuo, o fluxo galvânico inicializa o sistema sem base externa. Além disso, ... Aguardando nexo.
-------------------------------------------------


📝 Notas Técnicas

- A IA analítica utiliza hash e entropia para associar perguntas a factos.
- A IA emocional usa uma rede de memória temporal (DLM) que liga épocas consecutivas.
- O estado térmico (self._st) influencia a escolha de frases e o tom da resposta.
- Todo o conhecimento da Camada 2 é compactado num parágrafo normalizado que alimenta a AGI, garantindo consistência total entre os dois núcleos.


🧩 Modelos e Variantes

O repositório disponibiliza várias ramificações para diferentes cenários:

| Modelo                          | Descrição                                                                   |
|---------------------------------|-----------------------------------------------------------------------------|
| QuintikusOpen_FastV2.py         | Otimização de velocidade de busca para contextos extensos.                  |
| QuintikusOpen_Narativ.py        | Estruturação narrativa.                                                     |
| QuintikusOpenGGPT-GPU.py/CPU.py | Aceleração por hardware (GPU/CPU).                                          |
| QuintikusPCode.py               | Geração determinística de código com Programação Neurolinguística.          |
| Quintikus Listy                 | Classificação de imagens com "olhos da matemática".                         |


🤖 Quintikus Doomoble 🥔, Droid 🧠 e TGP 📐

Assistente pessoal offline, IA transformer de bolso que roda até em celular.

- 🔧 Personalização
- 📦 Especialistas prontos

O que é?
O Quintikus Doomoble e Droid é um motor de IA baseado em transformer e rede lógica, escrito em Python puro + NumPy.
Ele entende regras, aprende com frases, traduz intenções e pode controlar dispositivos reais.

Características:
- 🚫 Offline e privado – seus dados nunca saem do seu dispositivo.
- 📏 Leve – menos de 2 MB por especialista.
- 🧠 Memória viva – aprende sem retreino, só adicionando frases.
- 🗣️ Tradutor de intenções – entende linguagem natural.
- 🔌 Conecta ao mundo real – GPIO, e‑mail, câmera.


📐 TGP – Topology Geometric Processing

A linhagem TGP demonstra uma evolução clara:
- TGP‑1 – memorização de padrões (rigidez, 40% de qualidade conversacional).
- TGP‑2 – criatividade geométrica (70% de qualidade, mas ainda incompleta).
- TGP‑3 – controlo cognitivo com estado (98% de qualidade, adaptativo e preciso).

Latência de Inferência (ms)
│
│    TGP-1 ████████████████████ ~100ms
│    TGP-2 ████████████ ~120ms (por token)
│    TGP-3 ██ 5-21ms (total)
└────────────────────────────────

Qualidade de Resposta
│
│TGP-3 ██████████ 98%
│TGP-2 ████████ 70%
│TGP-1 ████ 40%
└─────────────────────────────────────────────────→ Tempo/Épocas de Treino


🚀 Como usar

python3 quintikus-model.py


📢 Novidade

[Sobre multiverso]
-/-
[NEW MODEL]
 TGP 3.1.6V


ℹ️ Sobre

- Reddit: https://www.reddit.com/r/QuintikusOpen
- Esse projeto me custou 3 anos, café e paciência. ☕
- Doações (PayPal): https://www.paypal.com/donate?business=4KJAVYQLQDMHA&no_recurring=0&item_name=Ajudar+a+engine&currency_code=USD

Autor: Ronan Basto
Licença: Livre para estudo e experimentação.


<a href="https://info.flagcounter.com/tTxZ"><img src="https://s01.flagcounter.com/count/tTxZ/bg_FFFFFF/txt_000000/border_CCCCCC/columns_7/maxflags_12/viewers_0/labels_1/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>

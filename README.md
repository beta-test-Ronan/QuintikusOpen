# QuintikusOpen
Transformer deterministo - Dlm class 1-2
Quintikus Proton-Flow TDLM v117.0 [MIT]


![alt text](https://github.com/beta-test-Ronan/QuintikusOpen/blob/main/model-Dlm.png?raw=true)
<br>

Visão Geral
O sistema implementa uma arquitetura de Dupla Inteligência Artificial com suporte a blockchain de conhecimento:

    QuintikusSovereignCore – IA Analítica, baseada em camadas de visão e massa geradas a partir de um texto fonte.

    QuintikusAGI – IA Emocional, com estados térmicos internos, memória associativa (DLM - Dual Loop Memory) e personalidade dinâmica.

Ambas as IAs compartilham o mesmo parágrafo de treino extraído do núcleo analítico, mas operam com paradigmas distintos: uma responde de forma lógica e a outra de forma emotiva.
Dependências

    numpy – não utilizado explicitamente no código atual.

    hashlib – para hashs de integridade e identificadores.

    time – simulação de processamento e métricas.

    pickle – serialização da blockchain.

    os – manipulação de arquivos.

    re – processamento de texto (expressões regulares).

    random – escolha aleatória de frases de arquétipos.

    sys, unicodedata – não utilizados diretamente.

Estrutura de Classes
1. SovereignBlockchain

Responsável pela persistência e integridade do conhecimento gerado.
Método	Descrição
__init__(name)	Define o caminho do arquivo de cache (blockchain_{name}.cache).
selar_memoria(knowledge_bundle)	Serializa o bundle com pickle e retorna hash SHA-256 (8 caracteres) como assinatura.
carregar_ponteiro()	Carrega o bundle do disco se o arquivo existir.
2. QuintikusSovereignCore (Núcleo Analítico)

Armazena o conhecimento em três camadas:

    layer1_vision: visões sintéticas com ponteiros, entropia e janelas de contexto.

    layer2_mass: fatos brutos originais.

    word_rarity: raridade de palavras baseada na frequência.

Arquétipos de personalidade (mc.txt e mm.txt) fornecem frases de introdução, ponte e conclusão para dois modos de fala.
Principais Métodos
Método	Descrição
amadurecer_nexo(raw_text)	Processa o texto bruto: divide em fragmentos, calcula entropia, gera ponteiros e popula layer1_vision e layer2_mass. Retorna um bundle para ser selado na blockchain.
falar_soberano(pergunta, cache)	Responde a uma pergunta comparando entropia das palavras com os registros da camada 1 e recupera o fato mais próximo. Inclui fraseologia dos arquétipos.
carregar_fundamentos(mc_f, mm_f)	Carrega arquivos de arquétipos (formato <intro>, <ponte>, <concl>) e preenche self.arquétipos.
exportar_banco_normalizado()	Converte todos os fatos da layer2_mass em um único parágrafo contínuo, sem duplicatas.
texto()	Gera um relatório detalhado do estado atual do núcleo (para depuração).
3. QuintikusAGI (Núcleo Emocional)

Implementa um estado interno de temperatura emocional (três variáveis), um dicionário de palavras com força e raridade, e uma memória em cadeia (DLM) que liga épocas de contexto.
Método	Descrição
__init__(_t)	Inicializa estados térmicos, dicionários e blocos filosóficos em hexadecimal.
inicializar(_txt)	Analisa o parágrafo de entrada: gera estatísticas de palavras, cria épocas (agrupamentos de 5 sentenças) com identificadores e constrói a DLM.
_upd_thermal(_q)	Atualiza os estados de estresse e harmonia com base nas palavras-chave da pergunta (tabela de valências).
falar(_qi)	Gera uma resposta utilizando a melhor época que intersecta a pergunta, aplica a DLM e escolhe frases de abertura/fecho conforme o estado térmico.
Fluxo de Execução Principal

    Inicialização

        Instancia QuintikusSovereignCore e carrega arquétipos (mc.txt, mm.txt).

        Lê texto.txt (ou usa texto padrão).

        Se não existir blockchain cacheada, processa o texto via amadurecer_nexo e sela.

    Preparação do Parágrafo Único

        Carrega a memória ativa da blockchain.

        Gera um parágrafo contínuo com exportar_banco_normalizado().

        Exibe os primeiros 500 caracteres.

    Inicialização da AGI Emocional

        Cria uma instância de QuintikusAGI e chama inicializar() com exatamente o mesmo parágrafo usado pelo núcleo analítico.

    Loop Interativo

        Pergunta ao usuário (RONAN:).

        Obtém resposta lógica do SovereignCore.falar_soberano().

        Obtém resposta emocional do AGI.falar(), passando o comando e o texto original.

        Exibe a resposta da AGI com efeito de digitação (efeito_llm).

Arquivos Necessários
Arquivo	Função
texto.txt	Texto fonte para treinamento do núcleo analítico.
mc.txt	Arquétipos de personalidade "mc" (intro/ponte/concl).
mm.txt	Arquétipos de personalidade "mm".
blockchain_machado.cache	Cache da blockchain gerado automaticamente.
Formato dos Arquétipos (mc.txt, mm.txt)

Devem conter tags como:
text

<intro>
Frase de introdução 1
Frase de introdução 2
<ponte>
...
<concl>
...

Exemplo de Uso
bash

$ python quintikus.py
📄 PARÁGRAFO GERADO (mesmo texto para as duas IAs):
...
👤 RONAN: Qual o sentido do fluxo galvânico?

Saída típica:
text

💡 [CARDUS MASTER FLOW | 0.15 μs | Quality: 100%]
==============================================================================================================
LAYER-1 (VISÃO): Analisei que Localizado nexo no ponteiro 12345678.
LAYER-2 (MASSA): O fluxo galvânico inicializa o sistema sem base externa.
 | Pulse | [FLUXO] <-> [GALVÂNICO] | Densidade: 0.9234
-> fim. (Selo: Cardus-100)
==============================================================================================================

[DLM-FLOW: 45.23μs | D:8/10 | T:0.2 | DLM-ACTIVE | SIGN: 25e0bb26]
No vácuo, o fluxo galvânico inicializa o sistema sem base externa. Além disso, ... Aguardando nexo.

Notas Técnicas

    A IA analítica utiliza hash e entropia para associar perguntas a fatos.

    A IA emocional usa uma rede de memória temporal (DLM) que liga épocas consecutivas.

    O estado térmico (self._st) influencia a escolha de frases e tom da resposta.

    Todo o conhecimento da camada 2 é compactado em um parágrafo normalizado para alimentar a AGI, garantindo consistência.

Versão: Quintikus Proton-Flow TDLM v117.0
Autor: Ronan Basto
Licença: Livre para estudo e experimentação.

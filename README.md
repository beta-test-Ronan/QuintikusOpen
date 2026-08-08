<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="google-site-verification" content="0RhHVcGc9hJucKrsUoyJWZ0QlK09-kCzY7InQTfsNZk" />
    <title>QuintikusOpen — Dual AI Architecture</title>
    <style>
        /* Reset básico e tipografia */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Roboto, system-ui, sans-serif;
            background: #f8f9fc;
            color: #1e1e2f;
            line-height: 1.6;
            padding: 2rem 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3, h4, h5 {
            color: #0b2b4a;
            margin-top: 1.8rem;
            margin-bottom: 0.8rem;
            border-bottom: 2px solid #d0d7e6;
            padding-bottom: 0.3rem;
        }
        h1 { font-size: 2.2rem; border-bottom: 4px solid #2a6f8f; }
        h2 { font-size: 1.8rem; }
        h3 { font-size: 1.4rem; border-bottom: 1px solid #cbd5e1; }
        h4 { font-size: 1.2rem; border-bottom: none; }
        h5 { font-size: 1.1rem; border-bottom: none; }
        p, li { margin-bottom: 0.5rem; }
        a { color: #1a6b8a; text-decoration: none; }
        a:hover { text-decoration: underline; }

        /* Container e cards */
        .container { background: #ffffff; border-radius: 16px; padding: 2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
        hr { border: none; border-top: 2px dashed #d0d7e6; margin: 2rem 0; }

        /* Tabelas */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.2rem 0;
            background: #fcfcfd;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }
        th {
            background: #e6edf5;
            color: #0b2b4a;
            font-weight: 600;
            padding: 0.8rem 1rem;
            text-align: left;
        }
        td {
            padding: 0.7rem 1rem;
            border-bottom: 1px solid #e9ecf0;
        }
        tr:last-child td { border-bottom: none; }
        code {
            background: #eef2f7;
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            font-size: 0.9em;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            color: #1f3a5f;
        }
        pre {
            background: #1e2636;
            color: #e3e8f0;
            padding: 1.2rem;
            border-radius: 12px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 1.2rem 0;
        }
        pre code {
            background: transparent;
            padding: 0;
            color: inherit;
            font-size: inherit;
        }
        blockquote {
            background: #f0f4fa;
            border-left: 5px solid #2a6f8f;
            padding: 1rem 1.5rem;
            margin: 1.2rem 0;
            border-radius: 0 12px 12px 0;
            font-style: italic;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .badge {
            display: inline-block;
            background: #1e3b5c;
            color: white;
            padding: 0.2rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .flag-counter {
            margin-top: 2rem;
            text-align: center;
        }
        .footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 2px solid #dce3ec;
            font-size: 0.95rem;
            color: #3a4a5e;
            text-align: center;
        }
        .translate-links {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem 1.2rem;
            justify-content: center;
            background: #eef2f7;
            padding: 0.8rem 1.5rem;
            border-radius: 40px;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }
        .translate-links a {
            color: #1f4b6e;
            font-weight: 500;
        }
        .translate-links a:hover { color: #0b2b4a; }
        ul, ol { padding-left: 1.8rem; }
        .diagram {
            background: #f1f5fb;
            padding: 1.2rem;
            border-radius: 12px;
            font-family: 'JetBrains Mono', monospace;
            white-space: pre;
            overflow-x: auto;
            font-size: 0.9rem;
            line-height: 1.6;
            color: #1f3a5f;
            border: 1px solid #d0dbe8;
        }
        /* Responsividade */
        @media (max-width: 700px) {
            body { padding: 1rem 0.5rem; }
            .container { padding: 1rem; }
            table { font-size: 0.9rem; }
            th, td { padding: 0.5rem; }
            .translate-links { gap: 0.3rem 0.8rem; }
        }
    </style>
</head>
<body>

<div class="container">

    <!-- Título principal -->
    <h1>🧠 QuintikusOpen — Dual AI Architecture with Knowledge Blockchain</h1>
    <p style="font-size: 1.2rem; color: #2a4b6a;">
        <strong>Transformer · LLM · CHATBOT · Topology Geometric Processing</strong><br>
        <span class="badge">Class Dlm 1‑2</span> 
        <span class="badge" style="background: #3d6a8c;">Quintikus [TDLM‑TGP‑SSML‑DLMC‑DSLM]</span>
        <span class="badge" style="background: #5f7d9c;">MIT</span>
    </p>

    <!-- Links de tradução -->
    <div class="translate-links">
        <strong>🌐 Traduzir:</strong>
        <a href="https://translate.google.com/translate?hl=en&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/">English</a>
        <a href="https://translate.google.com/translate?hl=es&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/">Español</a>
        <a href="https://translate.google.com/translate?hl=zh-CN&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/">中文 (简)</a>
        <a href="https://translate.google.com/translate?hl=zh-TW&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/">中文 (繁)</a>
        <a href="https://translate.google.com/translate?hl=af&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/">Africâner</a>
        <a href="https://translate.google.com/translate?hl=sw&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/">Swahili</a>
        <a href="https://translate.google.com/translate?hl=zu&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/">Zulu</a>
        <a href="https://translate.google.com/translate?hl=yo&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/">Yorùbá</a>
    </div>

    <!-- Imagem do modelo -->
    <img src="https://github.com/beta-test-Ronan/QuintikusOpen/blob/main/model-Dlm-f.png?raw=true" alt="Modelo Dual Loop Memory F">

    <hr>

    <!-- Visão Geral -->
    <h2>📌 Visão Geral</h2>
    <p>O sistema implementa uma arquitetura de <strong>Dupla Inteligência Artificial</strong> com suporte a <em>blockchain de conhecimento</em>:</p>
    <table>
        <thead>
            <tr><th>IA</th><th>Paradigma</th><th>Descrição</th></tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>QuintikusSovereignCore</strong></td>
                <td>Analítico / Lógico</td>
                <td>Baseado em camadas de visão e massa geradas a partir de um texto fonte.</td>
            </tr>
            <tr>
                <td><strong>QuintikusAGI</strong></td>
                <td>Emocional / Térmico</td>
                <td>Estados térmicos internos, memória associativa (DLM – <em>Dual Loop Memory</em>) e personalidade dinâmica.</td>
            </tr>
        </tbody>
    </table>
    <p>Ambas partilham o <strong>mesmo parágrafo de treino</strong> extraído do núcleo analítico, mas operam com paradigmas distintos:<br>
    uma responde de forma <strong>lógica</strong> e a outra de forma <strong>emotiva</strong>.</p>

    <hr>

    <!-- Dependências -->
    <h2>📦 Dependências</h2>
    <table>
        <thead><tr><th>Biblioteca</th><th>Utilização no código atual</th></tr></thead>
        <tbody>
            <tr><td><code>numpy</code></td><td>Processamento linear</td></tr>
            <tr><td><code>hashlib</code></td><td>Geração de hashes de integridade e identificadores</td></tr>
            <tr><td><code>time</code></td><td>Simulação de processamento e métricas</td></tr>
            <tr><td><code>pickle</code></td><td>Serialização da blockchain</td></tr>
            <tr><td><code>os</code></td><td>Manipulação de arquivos</td></tr>
            <tr><td><code>re</code></td><td>Processamento de texto (expressões regulares)</td></tr>
            <tr><td><code>random</code></td><td>Escolha aleatória de frases dos arquétipos</td></tr>
            <tr><td><code>sys</code>, <code>unicodedata</code></td><td>Configuração do sistema</td></tr>
        </tbody>
    </table>

    <hr>

    <!-- Estrutura de Classes -->
    <h2>🧱 Estrutura de Classes</h2>
    <div class="diagram">
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
    </div>

    <h3>1. SovereignBlockchain</h3>
    <p>Responsável pela persistência e integridade do conhecimento gerado.</p>
    <table>
        <thead><tr><th>Método</th><th>Descrição</th></tr></thead>
        <tbody>
            <tr><td><code>__init__(name)</code></td><td>Define o caminho do ficheiro de cache (<code>blockchain_{name}.cache</code>).</td></tr>
            <tr><td><code>selar_memoria(knowledge_bundle)</code></td><td>Serializa o <em>bundle</em> com <code>pickle</code> e retorna o hash SHA‑256 (8 caracteres) como assinatura.</td></tr>
            <tr><td><code>carregar_ponteiro()</code></td><td>Carrega o <em>bundle</em> do disco, se o ficheiro existir.</td></tr>
        </tbody>
    </table>

    <h3>2. QuintikusSovereignCore (Núcleo Analítico)</h3>
    <p>Armazena o conhecimento em três camadas:</p>
    <ul>
        <li><code>layer1_vision</code> – visões sintéticas com ponteiros, entropia e janelas de contexto.</li>
        <li><code>layer2_mass</code> – factos brutos originais.</li>
        <li><code>word_rarity</code> – raridade de palavras baseada na frequência.</li>
    </ul>

    <h4>🧠 Métodos Principais</h4>
    <table>
        <thead><tr><th>Método</th><th>Descrição</th></tr></thead>
        <tbody>
            <tr><td><code>amadurecer_nexo(raw_text)</code></td><td>Processa o texto bruto, divide‑o em fragmentos, calcula entropia e povoa <code>layer1_vision</code> e <code>layer2_mass</code>. Gera <em>bundle</em> para a blockchain.</td></tr>
            <tr><td><code>falar_soberano(pergunta, cache)</code></td><td>Gera respostas comparando a entropia da pergunta com a <strong>Camada 1</strong>. Usa os arquétipos para a saída textual.</td></tr>
            <tr><td><code>carregar_fundamentos(mc_f, mm_f)</code></td><td>Carrega os ficheiros de arquétipos e preenche as definições de personalidade.</td></tr>
            <tr><td><code>exportar_banco_normalizado()</code></td><td>Converte os factos da <code>layer2_mass</code> num parágrafo único e contínuo, livre de duplicados.</td></tr>
            <tr><td><code>texto()</code></td><td>Gera um relatório do estado do núcleo para depuração (<em>debug</em>).</td></tr>
        </tbody>
    </table>

    <blockquote>
        <strong>📖 Arquétipos de Personalidade</strong><br>
        Os ficheiros <code>mc.txt</code> e <code>mm.txt</code> definem o comportamento linguístico. São carregados no formato<br>
        <code>&lt;intro&gt;</code>, <code>&lt;ponte&gt;</code>, <code>&lt;concl&gt;</code> para popular o atributo <code>self.arquetipos</code>.
    </blockquote>

    <p><strong>Modelos de Execução Específicos:</strong><br>
    O repositório disponibiliza ramificações como <code>QuintikusOpen_FastV2.py</code> focado em otimização de velocidade de busca para contextos extensos; <code>QuintikusOpen_Narativ.py</code> voltado à estruturação narrativa e scripts direcionados para aceleração por hardware como <code>QuintikusOpenGGPT-GPU.py</code> e <code>QuintikusOpenGGPT-CPU.py</code>. A família Quintikus e o <code>QuintikusPCode.py</code> geram código determinístico com Programação Neurolinguística, e o <em>Quintikus Listy</em> vê com os olhos da matemática classificando imagens.</p>

    <h3>3. QuintikusAGI (Núcleo Emocional)</h3>
    <p>Implementa um <strong>estado interno de temperatura emocional</strong> (três variáveis), um dicionário de palavras com força e raridade, e uma memória em cadeia (<strong>DLM</strong>) que liga épocas de contexto.</p>
    <table>
        <thead><tr><th>Método</th><th>Descrição</th></tr></thead>
        <tbody>
            <tr><td><code>__init__(_t)</code></td><td>Inicializa estados térmicos, dicionários e blocos filosóficos em hexadecimal.</td></tr>
            <tr><td><code>inicializar(_txt)</code></td><td>Analisa o parágrafo de entrada: estatísticas de palavras, criação de épocas (grupos de 5 frases) e construção da DLM.</td></tr>
            <tr><td><code>_upd_thermal(_q)</code></td><td>Atualiza os estados de estresse e harmonia com base nas palavras‑chave da pergunta (tabela de valências).</td></tr>
            <tr><td><code>falar(_qi)</code></td><td>Gera uma resposta usando a melhor época que intersecta a pergunta, aplica a DLM e escolhe frases de abertura/fecho conforme o estado térmico.</td></tr>
        </tbody>
    </table>

    <hr>

    <!-- Fluxo -->
    <h2>🔁 Fluxo de Execução Principal</h2>
    <ol>
        <li><strong>Inicialização</strong>
            <ul>
                <li>Instancia <code>QuintikusSovereignCore</code> e carrega arquétipos (<code>mc.txt</code>, <code>mm.txt</code>).</li>
                <li>Lê <code>texto.txt</code> (ou usa texto padrão).</li>
                <li>Se não existir blockchain em cache, processa o texto via <code>amadurecer_nexo</code> e sela.</li>
            </ul>
        </li>
        <li><strong>Preparação do Parágrafo Único</strong>
            <ul>
                <li>Carrega a memória ativa da blockchain.</li>
                <li>Gera um parágrafo contínuo com <code>exportar_banco_normalizado()</code>.</li>
                <li>Exibe os primeiros 500 caracteres.</li>
            </ul>
        </li>
        <li><strong>Inicialização da AGI Emocional</strong>
            <ul>
                <li>Cria uma instância de <code>QuintikusAGI</code> e chama <code>inicializar()</code> com o mesmo parágrafo usado pelo núcleo analítico.</li>
            </ul>
        </li>
        <li><strong>Loop Interativo</strong>
            <ul>
                <li>Pergunta ao utilizador (<code>RONAN:</code>).</li>
                <li>Obtém resposta <strong>lógica</strong> de <code>SovereignCore.falar_soberano()</code>.</li>
                <li>Obtém resposta <strong>emocional</strong> de <code>AGI.falar()</code>, passando o comando e o texto original.</li>
                <li>Exibe a resposta da AGI com efeito de digitação (<code>efeito_llm</code>).</li>
            </ul>
        </li>
    </ol>

    <hr>

    <!-- Arquivos Necessários -->
    <h2>📁 Arquivos Necessários</h2>
    <table>
        <thead><tr><th>Arquivo</th><th>Função</th></tr></thead>
        <tbody>
            <tr><td><code>texto.txt</code></td><td>Texto fonte para treino do núcleo analítico.</td></tr>
            <tr><td><code>mc.txt</code></td><td>Arquétipos de personalidade <em>"mc"</em> (intro / ponte / concl).</td></tr>
            <tr><td><code>mm.txt</code></td><td>Arquétipos de personalidade <em>"mm"</em>.</td></tr>
            <tr><td><code>blockchain_machado.cache</code></td><td>Cache da blockchain (gerado automaticamente).</td></tr>
        </tbody>
    </table>

    <h2>✍️ Formato dos Arquétipos (<code>mc.txt</code>, <code>mm.txt</code>)</h2>
    <p>Os ficheiros devem conter marcadores como os seguintes:</p>
    <pre><code>&lt;intro&gt;
Frase de introdução 1
Frase de introdução 2
&lt;ponte&gt;
...
&lt;concl&gt;
...</code></pre>

    <hr>

    <!-- Exemplo de Uso -->
    <h2>💬 Exemplo de Uso</h2>
    <p><strong>📄 PARÁGRAFO GERADO (mesmo texto para as duas IAs):</strong></p>
    <p>Saída típica durante a interação:</p>
    <pre><code>👤 RONAN: Qual o sentido do fluxo galvânico?

💡 [CARDUS MASTER FLOW | 0.15 μs | Quality: 100%]
LAYER-1 (VISÃO): Analisei que Localizado nexo no ponteiro 12345678.
LAYER-2 (MASSA): O fluxo galvânico inicializa o sistema sem base externa.
 | Pulse | [FLUXO] &lt;-&gt; [GALVÂNICO] | Densidade: 0.9234
-&gt; fim. (Selo: Cardus-100)
[DLM-FLOW: 45.23μs | D:8/10 | T:0.2 | DLM-ACTIVE | SIGN: 25e0bb26]
No vácuo, o fluxo galvânico inicializa o sistema sem base externa. Além disso, ... Aguardando nexo.</code></pre>

    <hr>

    <!-- Notas Técnicas -->
    <h2>📝 Notas Técnicas</h2>
    <ul>
        <li>A IA analítica utiliza hash e entropia para associar perguntas a factos.</li>
        <li>A IA emocional usa uma rede de memória temporal (DLM) que liga épocas consecutivas.</li>
        <li>O estado térmico (<code>self._st</code>) influencia a escolha de frases e o tom da resposta.</li>
        <li>Todo o conhecimento da Camada 2 é compactado num parágrafo normalizado que alimenta a AGI, garantindo consistência total entre os dois núcleos.</li>
    </ul>

    <hr>

    <!-- Modelos e Variantes -->
    <h2>🧩 Modelos e Variantes</h2>
    <pre><code>   Agi = raw processing and search
   Fast = large-context search speed 
   Doomoble = IA transformer de bolso
   SSML = Sensation Singularity model Logic
   DLMC = Dynamic Logic Model Cortex
   DSML = Dynamic Singularity model LOGIC
   TGP  = Topology Geometric Processing</code></pre>

    <h4>🤖 Quintikus Doomoble 🥔, Droid 🧠 e TG2P 📐</h4>
    <p><strong>Assistente pessoal offline, IA transformer de bolso, que roda até em celular.</strong></p>
    <ul>
        <li>🔧 Personalização</li>
        <li>📦 Especialistas prontos</li>
    </ul>

    <h4>O que é?</h4>
    <p>O Quintikus Doomoble e Droid é um motor de IA baseado em transformer e rede lógica, escrito em Python puro + NumPy.<br>
    Ele entende regras, aprende com frases, traduz intenções e pode controlar dispositivos reais.</p>

    <h4>Características</h4>
    <ul>
        <li>🚫 <strong>Offline e privado</strong> – seus dados nunca saem do seu dispositivo.</li>
        <li>📏 <strong>Leve</strong> – menos de 2 MB por especialista.</li>
        <li>🧠 <strong>Memória viva</strong> – aprende sem retreino, só adicionando frases.</li>
        <li>🗣️ <strong>Tradutor de intenções</strong> – entende linguagem natural.</li>
        <li>🔌 <strong>Conecta ao mundo real</strong> – GPIO, e‑mail, câmera.</li>
    </ul>

    <h5>TGP: Topology Geometric Processing</h5>
    <p>A linhagem TGP demonstra uma evolução clara desde a memorização de padrões (TGP‑1) até à criatividade geométrica (TGP‑2) e, finalmente, ao controlo cognitivo com estado (TGP‑3). A arquitetura ARQUINET não é uma melhoria incremental, mas um salto geracional, oferecendo precisão e velocidade para implantação no mundo real.</p>

    <pre><code>    Latência de Inferência (ms)
    │
    │    TGP-1 ████████████████████ ~100ms
    │    TGP-2 ████████████ ~120ms (por token)
    │    TGP-3 ██ 5-21ms (total)
    │
    └────────────────────────────────

    Qualidade de Resposta
    │
    │TGP-3 ██████████ 98%
    │                    
    │TGP-2 ████████ 70%
    │          
    │TGP-1 ████ 40%
    │    
    └─────────────────────────────────────────────────→ Tempo/Épocas de Treino
    
    Conversational Quality  
    │ TGP‑1  ████████ 40% (rigid, copied)
    │ TGP‑2  ██████████████ 70% (creative but unfinished)
    │ TGP‑3  ████████████████████ 98% (precise, adaptive)
    └────────────────────────────────────────────────→ higher is better</code></pre>

    <h2>🚀 Como usar</h2>
    <pre><code>python3 quintikus-model.py</code></pre>

    <hr>

    <!-- Novidade -->
    <h4>📢 Novidade</h4>
    <pre><code>    [Sobre multiverso]
    -/-
    [NEW MODEL]
     TGP 3.1.6V</code></pre>

    <hr>

    <!-- Rodapé / Sobre -->
    <div class="footer">
        <p>
            <strong>Reddit:</strong> <a href="https://www.reddit.com/r/QuintikusOpen">r/QuintikusOpen</a><br>
            Esse projeto me custou 3 anos, café e paciência. ☕<br>
            <strong>Donate (PayPal):</strong> <a href="https://www.paypal.com/donate?business=4KJAVYQLQDMHA&no_recurring=0&item_name=Ajudar+a+engine&currency_code=USD">Ajudar a engine</a>
        </p>
        <p><strong>Autor:</strong> Ronan Basto<br>
        <strong>Licença:</strong> Livre para estudo e experimentação.</p>
    </div>

    <!-- Flag Counter -->
    <div class="flag-counter">
   <a href="https://info.flagcounter.com/tTxZ"><img src="https://s01.flagcounter.com/count/tTxZ/bg_FFFFFF/txt_000000/border_CCCCCC/columns_7/maxflags_12/viewers_0/labels_1/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
    </div>

</div><!-- fim container -->

</body>
</html>

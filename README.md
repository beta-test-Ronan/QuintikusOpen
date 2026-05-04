🌐 **Traduzir:**  
[English](https://translate.google.com/translate?hl=en&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) | [Español](https://translate.google.com/translate?hl=es&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) | [中文 (简)](https://translate.google.com/translate?hl=zh-CN&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) | [中文 (繁)](https://translate.google.com/translate?hl=zh-TW&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/)  
[Africâner](https://translate.google.com/translate?hl=af&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) | [Swahili](https://translate.google.com/translate?hl=sw&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) | [Zulu](https://translate.google.com/translate?hl=zu&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/) | [Yorùbá](https://translate.google.com/translate?hl=yo&sl=pt&u=https://beta-test-ronan.github.io/QuintikusOpen/)
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="google-site-verification" content="CIF8gF5LWM_CsvmLmaJKaJmpzZS34aOlwjrQzks4LDo" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-PT5BVGX8"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->
</head>
<body id="google_translate_element">

<p><strong>Transformer deterministo – Dlm class 1‑2</strong><br>
<strong>Quintikus Proton-Flow TDLM v117.0 [MIT]</strong></p>

<img src="https://github.com/beta-test-Ronan/QuintikusOpen/blob/main/model-Dlm-f.png?raw=true" alt="Modelo Dual Loop Memory F">

<hr>

<h2>📌 Visão Geral</h2>

<p>O sistema implementa uma arquitetura de <strong>Dupla Inteligência Artificial</strong> com suporte a blockchain de conhecimento:</p>

<ul>
    <li><strong>QuintikusSovereignCore</strong> – IA Analítica, baseada em camadas de visão e massa geradas a partir de um texto fonte.</li>
    <li><strong>QuintikusAGI</strong> – IA Emocional, com estados térmicos internos, memória associativa (DLM – <em>Dual Loop Memory</em>) e personalidade dinâmica.</li>
</ul>

<p>Ambas as IAs partilham o mesmo parágrafo de treino extraído do núcleo analítico, mas operam com paradigmas distintos:<br>
uma responde de forma <strong>lógica</strong> e a outra de forma <strong>emotiva</strong>.</p>

<hr>

<h2>📦 Dependências</h2>

<table>
    <thead>
        <tr>
            <th>Biblioteca</th>
            <th>Utilização no código atual</th>
        </tr>
    </thead>
    <tbody>
        <tr><td><code>numpy</code></td><td>Processamento linear</td></tr>
        <tr><td><code>hashlib</code></td><td>Geração de hashes de integridade e identificadores</td></tr>
        <tr><td><code>time</code></td><td>Simulação de processamento e métricas</td></tr>
        <tr><td><code>pickle</code></td><td>Serialização da blockchain</td></tr>
        <tr><td><code>os</code></td><td>Manipulação de arquivos</td></tr>
        <tr><td><code>re</code></td><td>Processamento de texto (expressões regulares)</td></tr>
        <tr><td><code>random</code></td><td>Escolha aleatória de frases dos arquétipos</td></tr>
        <tr><td><code>sys</code>, <code>unicodedata</code></td><td>Config system</td></tr>
    </tbody>
</table>

<h2>⚙️ Estrutura de Classes</h2>

<h3>1. SovereignBlockchain</h3>
<p>Responsável pela persistência e integridade do conhecimento gerado.</p>

<table>
    <thead>
        <tr><th>Método</th><th>Descrição</th></tr>
    </thead>
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
    <thead>
        <tr><th>Método</th><th>Descrição</th></tr>
    </thead>
    <tbody>
        <tr><td><code>amadurecer_nexo(raw_text)</code></td><td>Processa o texto bruto, divide‑o em fragmentos, calcula entropia e povoa <code>layer1_vision</code> e <code>layer2_mass</code>. Gera <em>bundle</em> para a blockchain.</td></tr>
        <tr><td><code>falar_soberano(pergunta, cache)</code></td><td>Gera respostas comparando a entropia da pergunta com a <strong>Camada 1</strong>. Usa os arquétipos para a saída textual.</td></tr>
        <tr><td><code>carregar_fundamentos(mc_f, mm_f)</code></td><td>Carrega os ficheiros de arquétipos e preenche as definições de personalidade.</td></tr>
        <tr><td><code>exportar_banco_normalizado()</code></td><td>Converte os factos da <code>layer2_mass</code> num parágrafo único e contínuo, livre de duplicados.</td></tr>
        <tr><td><code>texto()</code></td><td>Gera um relatório do estado do núcleo para depuração (<em>debug</em>).</td></tr>
    </tbody>
</table>

<blockquote>
<p>📖 <strong>Arquétipos de Personalidade</strong><br>
Os ficheiros <code>mc.txt</code> e <code>mm.txt</code> definem o comportamento linguístico. São carregados no formato<br>
<code>&lt;intro&gt;</code>, <code>&lt;ponte&gt;</code>, <code>&lt;concl&gt;</code> para popular o atributo <code>self.arquetipos</code>.</p>
</blockquote>

<h3>3. QuintikusAGI (Núcleo Emocional)</h3>
<p>Implementa um <strong>estado interno de temperatura emocional</strong> (três variáveis), um dicionário de palavras com força e raridade, e uma memória em cadeia (<strong>DLM</strong>) que liga épocas de contexto.</p>

<table>
    <thead>
        <tr><th>Método</th><th>Descrição</th></tr>
    </thead>
    <tbody>
        <tr><td><code>__init__(_t)</code></td><td>Inicializa estados térmicos, dicionários e blocos filosóficos em hexadecimal.</td></tr>
        <tr><td><code>inicializar(_txt)</code></td><td>Analisa o parágrafo de entrada: estatísticas de palavras, criação de épocas (grupos de 5 frases) e construção da DLM.</td></tr>
        <tr><td><code>_upd_thermal(_q)</code></td><td>Atualiza os estados de estresse e harmonia com base nas palavras‑chave da pergunta (tabela de valências).</td></tr>
        <tr><td><code>falar(_qi)</code></td><td>Gera uma resposta usando a melhor época que intersecta a pergunta, aplica a DLM e escolhe frases de abertura/fecho conforme o estado térmico.</td></tr>
    </tbody>
</table>

<hr>

<h2>🔁 Fluxo de Execução Principal</h2>

<ol>
    <li>
        <strong>Inicialização</strong>
        <ul>
            <li>Instancia <code>QuintikusSovereignCore</code> e carrega arquétipos (<code>mc.txt</code>, <code>mm.txt</code>).</li>
            <li>Lê <code>texto.txt</code> (ou usa texto padrão).</li>
            <li>Se não existir blockchain em cache, processa o texto via <code>amadurecer_nexo</code> e sela.</li>
        </ul>
    </li>
    <li>
        <strong>Preparação do Parágrafo Único</strong>
        <ul>
            <li>Carrega a memória ativa da blockchain.</li>
            <li>Gera um parágrafo contínuo com <code>exportar_banco_normalizado()</code>.</li>
            <li>Exibe os primeiros 500 caracteres.</li>
        </ul>
    </li>
    <li>
        <strong>Inicialização da AGI Emocional</strong>
        <ul>
            <li>Cria uma instância de <code>QuintikusAGI</code> e chama <code>inicializar()</code> com o mesmo parágrafo usado pelo núcleo analítico.</li>
        </ul>
    </li>
    <li>
        <strong>Loop Interativo</strong>
        <ul>
            <li>Pergunta ao utilizador (<code>RONAN:</code>).</li>
            <li>Obtém resposta <strong>lógica</strong> de <code>SovereignCore.falar_soberano()</code>.</li>
            <li>Obtém resposta <strong>emocional</strong> de <code>AGI.falar()</code>, passando o comando e o texto original.</li>
            <li>Exibe a resposta da AGI com efeito de digitação (<code>efeito_llm</code>).</li>
        </ul>
    </li>
</ol>

<hr>

<h2>📁 Arquivos Necessários</h2>

<table>
    <thead>
        <tr><th>Arquivo</th><th>Função</th></tr>
    </thead>
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
...
</code></pre>

<h2>💬 Exemplo de Uso</h2>

<pre><code>📄 PARÁGRAFO GERADO (mesmo texto para as duas IAs):
</code></pre>

<p>Saída típica durante a interação:</p>

<pre><code class="language-text">👤 RONAN: Qual o sentido do fluxo galvânico?

💡 [CARDUS MASTER FLOW | 0.15 μs | Quality: 100%]
LAYER-1 (VISÃO): Analisei que Localizado nexo no ponteiro 12345678.
LAYER-2 (MASSA): O fluxo galvânico inicializa o sistema sem base externa.
 | Pulse | [FLUXO] &lt;-&gt; [GALVÂNICO] | Densidade: 0.9234
-&gt; fim. (Selo: Cardus-100)
[DLM-FLOW: 45.23μs | D:8/10 | T:0.2 | DLM-ACTIVE | SIGN: 25e0bb26]
No vácuo, o fluxo galvânico inicializa o sistema sem base externa. Além disso, ... Aguardando nexo.
</code></pre>

<h2>📝 Notas Técnicas</h2>

<ul>
    <li>A IA analítica utiliza hash e entropia para associar perguntas a factos.</li>
    <li>A IA emocional usa uma rede de memória temporal (DLM) que liga épocas consecutivas.</li>
    <li>O estado térmico (<code>self._st</code>) influencia a escolha de frases e o tom da resposta.</li>
    <li>Todo o conhecimento da Camada 2 é compactado num parágrafo normalizado que alimenta a AGI, garantindo consistência total entre os dois núcleos.</li>
</ul>

<h2>Model</h2>

<pre><code>   Agi = precessamento bruto e busca&lt;br&gt;
   Fast = velocidade de busca grande contexto 
   Doomoble = IA transformer de bolso
</code></pre>

<h1>Quintikus Doomoble 🥔🧠</h1>

<p><strong>Assistente pessoal offline, IA transformer de bolso, que roda até em celular.</strong></p>

<ul>
    <li>📜 Código Principal</li>
    <li>🧠 Como usar</li>
    <li>🔧 Personalização</li>
    <li>📦 Especialistas prontos</li>
    <li>📄 Licença MIT<</li>
</ul>

<h2>O que é?</h2>
<p>O Quintikus Doomoble é um motor de IA baseado em transformer, escrito em Python puro + NumPy.<br>
Ele entende regras, aprende com frases, traduz intenções, e pode controlar dispositivos reais.</p>

<h2>Características</h2>
<ul>
    <li>🚫 <strong>Offline e privado</strong> – seus dados nunca saem do seu dispositivo.</li>
    <li>📏 <strong>Leve</strong> – menos de 2 MB por especialista.</li>
    <li>🧠 <strong>Memória viva</strong> – aprende sem retreino, só adicionando frases.</li>
    <li>🗣️ <strong>Tradutor de intenções</strong> – entende linguagem natural.</li>
    <li>🔌 <strong>Conecta ao mundo real</strong> – GPIO, e‑mail, câmera.</li>
</ul>

<h2>Como usar</h2>
<pre><code>python3 QuintikusOpenDoomoble.py
</code></pre>

<p>Versão: Quintikus Proton-Flow TDLM v117.0<br>
Autor: Ronan Basto<br>
Licença: Livre para estudo e experimentação.</p>

</body>
</html>

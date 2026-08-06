import numpy as np
import re
import pickle
from collections import defaultdict

# ============================================================
# CLASSE TGP-1
# ============================================================

class TGP:
    """
    TGP-1 - Transpassagem Geométrica de Propagação
    Com persistência (pickle) e fine-tuning.
    """
    
    def __init__(self, dim_espaco=64, num_niveis=3):
        self.dim_espaco = dim_espaco
        self.num_niveis = num_niveis
        
        # Níveis topológicos
        self.niveis = []
        for i in range(num_niveis):
            nivel = {
                'transformacao': np.random.randn(dim_espaco, dim_espaco) * 0.01,
                'curvatura': np.random.randn(dim_espaco) * 0.001,
            }
            self.niveis.append(nivel)
        
        # Dicionários de tokens
        self.token_para_coord = {}
        self.coord_para_token = {}
        self.token_fim = '<END>'
        self.token_user = '<USER>'
        self.token_bot = '<BOT>'
        
        # Memória de episódios
        self.episodios = []         # lista de dicts
        self.historico_curvas = []
        
        # Estatísticas
        self.total_tokens = 0
        
        # Bigramas (defaultdict serializável com pickle)
        self.bigramas = defaultdict(lambda: defaultdict(int))
        self.total_bigramas = 0
        
        # Registrar tokens especiais
        self._registrar_token(self.token_fim)
        self._registrar_token(self.token_user)
        self._registrar_token(self.token_bot)
    
    def _registrar_token(self, token):
        if token not in self.token_para_coord:
            coord = np.random.randn(self.dim_espaco) * 0.1
            coord /= np.linalg.norm(coord) + 1e-8
            self.token_para_coord[token] = coord
            self.coord_para_token[len(self.coord_para_token)] = token
            self.total_tokens += 1
    
    # ---------- Tokenização ----------
    def tokenizar(self, texto):
        tokens = re.findall(r'[a-záéíóúâêîôûãõç]+|[0-9]+|[.,!?;:]+|\s+', texto.lower())
        resultado = []
        for t in tokens:
            if t.isspace() and resultado:
                resultado[-1] += t
            elif t.strip():
                resultado.append(t)
        return resultado
    
    def tokens_para_texto(self, tokens):
        texto = ''.join(tokens)
        texto = texto.replace(self.token_fim, '')
        texto = texto.replace(self.token_user, '')
        texto = texto.replace(self.token_bot, '')
        return texto.strip()
    
    # ---------- Bigramas ----------
    def aprender_bigramas(self, tokens):
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i+1]
            self.bigramas[a][b] += 1
            self.total_bigramas += 1
    
    # ---------- Transpassagem ----------
    def transpassar(self, tokens, armazenar=True):
        if not tokens:
            return [], {}
        
        for t in tokens:
            self._registrar_token(t)
        
        curva = []
        for i, token in enumerate(tokens):
            ponto = self.token_para_coord[token].copy()
            
            for nivel in self.niveis:
                transformada = np.dot(ponto, nivel['transformacao'])
                transformada = np.clip(transformada, -3, 3)
                ponto = np.tanh(transformada)
                ponto += nivel['curvatura']
            
            norma = np.linalg.norm(ponto)
            if norma > 0:
                ponto /= norma
            
            peso_pos = 1.0 / (1.0 + 0.1 * abs(i - len(tokens)/2))
            ponto *= peso_pos
            curva.append(ponto)
        
        if len(curva) > 2:
            curva_suave = []
            for i in range(len(curva)):
                if i == 0:
                    p = curva[0] * 0.7 + curva[1] * 0.3
                elif i == len(curva) - 1:
                    p = curva[-1] * 0.7 + curva[-2] * 0.3
                else:
                    p = curva[i-1] * 0.2 + curva[i] * 0.6 + curva[i+1] * 0.2
                norma = np.linalg.norm(p)
                if norma > 0:
                    p /= norma
                curva_suave.append(p)
            curva = curva_suave
        
        propriedades = self._calcular_geometria(curva)
        
        if armazenar:
            self.historico_curvas.append(curva)
            if len(self.historico_curvas) > 20:
                self.historico_curvas.pop(0)
        
        return curva, propriedades
    
    def _calcular_geometria(self, curva):
        if len(curva) < 2:
            return {'centroide': curva[0] if curva else np.zeros(self.dim_espaco),
                    'direcao_principal': np.zeros(self.dim_espaco)}
        
        pontos = np.array(curva)
        centroide = np.mean(pontos, axis=0)
        
        if len(curva) >= 3:
            diffs = np.diff(pontos, axis=0)
            direcao = np.mean(diffs, axis=0)
            norma = np.linalg.norm(direcao)
            if norma > 0:
                direcao /= norma
        else:
            direcao = curva[-1] - curva[0]
            norma = np.linalg.norm(direcao)
            if norma > 0:
                direcao /= norma
        
        return {'centroide': centroide, 'direcao_principal': direcao, 'comprimento': len(curva)}
    
    # ---------- Ressonância ----------
    def ressonancia(self, curva_a, curva_b):
        if len(curva_a) < 2 or len(curva_b) < 2:
            if len(curva_a) > 0 and len(curva_b) > 0:
                return np.dot(curva_a[-1], curva_b[-1])
            return 0.0
        
        dirs_a, dirs_b = [], []
        for i in range(len(curva_a)-1):
            d = curva_a[i+1] - curva_a[i]
            n = np.linalg.norm(d)
            if n > 1e-8: dirs_a.append(d/n)
        for i in range(len(curva_b)-1):
            d = curva_b[i+1] - curva_b[i]
            n = np.linalg.norm(d)
            if n > 1e-8: dirs_b.append(d/n)
        
        if not dirs_a or not dirs_b:
            return 0.0
        
        similaridade = sum(np.dot(dirs_a[i], dirs_b[i]) for i in range(min(len(dirs_a), len(dirs_b))))
        similaridade /= min(len(dirs_a), len(dirs_b))
        
        cent_a = np.mean(curva_a, axis=0)
        cent_b = np.mean(curva_b, axis=0)
        sim_cent = np.dot(cent_a, cent_b)
        
        return 0.7 * similaridade + 0.3 * sim_cent
    
    # ---------- Aprendizado ----------
    def aprender_par(self, tokens_entrada, tokens_saida, taxa=0.01):
        for t in tokens_entrada + tokens_saida:
            self._registrar_token(t)
        
        curva_entrada, _ = self.transpassar(tokens_entrada, armazenar=False)
        curva_saida, _ = self.transpassar(tokens_saida, armazenar=False)
        if not curva_entrada or not curva_saida:
            return
        
        ponto_ent = curva_entrada[-1]
        ponto_sai = curva_saida[0]
        direcao = ponto_sai - ponto_ent
        
        for i, token in enumerate(tokens_saida):
            coord = self.token_para_coord[token]
            alvo = ponto_ent + direcao * (i+1)/len(tokens_saida)
            self.token_para_coord[token] = coord + taxa * (alvo - coord)
            norma = np.linalg.norm(self.token_para_coord[token])
            if norma > 0:
                self.token_para_coord[token] /= norma
        
        todos = tokens_entrada + tokens_saida
        for i in range(len(todos)):
            for j in range(i+1, len(todos)):
                if todos[i] != todos[j]:
                    ci = self.token_para_coord[todos[i]]
                    cj = self.token_para_coord[todos[j]]
                    self.token_para_coord[todos[i]] += 0.001 * taxa * (cj - ci)
                    n = np.linalg.norm(self.token_para_coord[todos[i]])
                    if n > 0: self.token_para_coord[todos[i]] /= n
        
        self.aprender_bigramas(tokens_entrada)
        self.aprender_bigramas(tokens_saida)
        
        self.episodios.append({
            'entrada': tokens_entrada,
            'saida': tokens_saida,
            'curva_entrada': curva_entrada,
            'curva_saida': curva_saida
        })
        if len(self.episodios) > 100:
            self.episodios.pop(0)
    
    # ---------- Geração ----------
    def gerar_resposta_v2(self, tokens_entrada, max_tokens=15, temperatura=0.5):
        curva_entrada, _ = self.transpassar(tokens_entrada, armazenar=True)
        if not curva_entrada:
            return [self.token_fim]
        
        # Episódio mais ressonante
        melhor_ep, melhor_res = None, -1
        for ep in self.episodios:
            if ep['curva_entrada']:
                res = self.ressonancia(curva_entrada, ep['curva_entrada'])
                if res > melhor_res:
                    melhor_res = res
                    melhor_ep = ep
        
        if melhor_ep and melhor_res > 0.3:
            resposta = []
            usados = set()
            for token in melhor_ep['saida']:
                if token in usados: continue
                if token in self.token_para_coord:
                    resposta.append(token)
                    usados.add(token)
                if len(resposta) >= max_tokens: break
            if resposta:
                return resposta
        
        # Fallback com bigramas
        campo = curva_entrada[-1].copy()
        resposta, usados, ultimo = [], set(), None
        ponto = campo.copy()
        
        for _ in range(max_tokens):
            candidatos = []
            for token, coord in self.token_para_coord.items():
                if token in [self.token_fim, self.token_user, self.token_bot] or token in usados:
                    continue
                sim = np.dot(ponto, coord)
                bonus = 0.0
                if ultimo and ultimo in self.bigramas and token in self.bigramas[ultimo]:
                    prob = self.bigramas[ultimo][token] / sum(self.bigramas[ultimo].values())
                    bonus = prob * 0.5
                candidatos.append((token, sim + bonus))
            
            if not candidatos: break
            candidatos.sort(key=lambda x: x[1], reverse=True)
            k = min(10, len(candidatos))
            top = candidatos[:k]
            scores = np.array([s[1] for s in top])
            scores = scores / (temperatura + 0.1)
            scores -= np.max(scores)
            probs = np.exp(scores)
            probs /= probs.sum()
            idx = np.random.choice(len(top), p=probs)
            token_escolhido = top[idx][0]
            
            resposta.append(token_escolhido)
            usados.add(token_escolhido)
            ultimo = token_escolhido
            coord_esc = self.token_para_coord[token_escolhido]
            ponto = ponto * 0.5 + coord_esc * 0.5
            ponto /= np.linalg.norm(ponto) + 1e-8
            if top[0][1] < 0.05: break
        
        return resposta
    
    def responder(self, texto_usuario):
        tokens = self.tokenizar(texto_usuario)
        if not tokens: return "..."
        resp = self.gerar_resposta_v2(tokens, max_tokens=15, temperatura=0.5)
        if resp:
            self.aprender_par(tokens, resp[:5], taxa=0.005)
        return self.tokens_para_texto(resp)
    
    # ========== PERSISTÊNCIA (PICKLE) ==========
    def salvar(self, arquivo):
        # Converte defaultdict para dict normal para pickle
        bigramas_serial = {a: dict(b) for a, b in self.bigramas.items()}
        estado = {
            'dim_espaco': self.dim_espaco,
            'num_niveis': self.num_niveis,
            'niveis': self.niveis,
            'token_para_coord': self.token_para_coord,
            'coord_para_token': self.coord_para_token,
            'episodios': self.episodios,
            'total_tokens': self.total_tokens,
            'bigramas': bigramas_serial,
            'total_bigramas': self.total_bigramas,
        }
        with open(arquivo, 'wb') as f:
            pickle.dump(estado, f)
        print(f"💾 Modelo salvo em {arquivo}")
    
    def carregar(self, arquivo):
        with open(arquivo, 'rb') as f:
            estado = pickle.load(f)
        
        self.dim_espaco = estado['dim_espaco']
        self.num_niveis = estado['num_niveis']
        self.niveis = estado['niveis']
        self.token_para_coord = estado['token_para_coord']
        self.coord_para_token = estado['coord_para_token']
        self.episodios = estado['episodios']
        self.total_tokens = estado['total_tokens']
        self.total_bigramas = estado['total_bigramas']
        
        # Reconstrói bigramas como defaultdict
        self.bigramas = defaultdict(lambda: defaultdict(int))
        bigramas_salvos = estado['bigramas']
        for a, bs in bigramas_salvos.items():
            for b, count in bs.items():
                self.bigramas[a][b] = count
        
        print(f"📂 Modelo carregado de {arquivo}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    print("🧠 Treinando TGP-1 base...")
    tgp = TGP(dim_espaco=128, num_niveis=2)

    # Dados de treino iniciais
    dialogos_base = [
        ("oi", "oi, tudo bem?"),
        ("tudo bem", "tudo ótimo!"),
        ("qual é o seu nome?", "meu nome é TGP!")
    
    ]

    for entrada, saida in dialogos_base:
        tokens_entrada = tgp.tokenizar(entrada)
        tokens_saida = tgp.tokenizar(saida)
        for _ in range(20):
            tgp.aprender_par(tokens_entrada, tokens_saida, taxa=0.02)

    print(f"   ✅ {tgp.total_tokens} tokens, {len(tgp.episodios)} episódios, {tgp.total_bigramas} bigramas")
    tgp.salvar("tgp1_base.pkl")

    # ----------------------------------------------------------
 
    print("\n🔄 Fine-tuning com novos dados...")
    tgp2 = TGP()
    tgp2.carregar("tgp1_base.pkl")

    novos_dialogos = [
        ("oi", "oi, tudo bem?"),
        ("tudo bem", "tudo ótimo!"),
        ("qual é o seu nome?", "meu nome é TGP!"),
        ("adeus", "entendi, tenha bom dia"),
        ("tchau", "entendi, tenha se cuida!")
        
    ]

    for entrada, saida in novos_dialogos:
        tokens_entrada = tgp2.tokenizar(entrada)
        tokens_saida = tgp2.tokenizar(saida)
        for _ in range(10):
            tgp2.aprender_par(tokens_entrada, tokens_saida, taxa=0.01)

    tgp2.salvar("tgp1_finetuned.pkl")

    # ----------------------------------------------------------
    print("\n✅ Testes após fine-tuning:")
    testes = [
        "oi",
        "bom dia",
        "tchau",
    ]
    for pergunta in testes:
        resp = tgp2.responder(pergunta)
        print(f"  🙋 {pergunta}")
        print(f"  🤖 {resp}\n")

    print("✅ Pronto! Modelos salvos: tgp1_base.pkl e tgp1_finetuned.pkl")

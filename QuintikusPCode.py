import math
import tkinter as tk
from tkinter import scrolledtext
import threading
import re
import os

# ============================================================
# 1. MOTOR NEURO-ENTRÓPICO
# ============================================================
neuro = [
    [1, 2, 3, 4, 5], 
    ["def", "print", "class", "dentro", "fora"]
]
pesos = [0.2] * 5
acao = []

def softmax(pontuacoes):
    max_val = max(pontuacoes)
    exp_val = [math.exp(p - max_val) for p in pontuacoes]
    return [v/sum(exp_val) for v in exp_val]

def ajustar_pesos(texto, dogma_alvo):
    global pesos
    texto_lower = texto.lower()
    ativacoes = [pesos[i] * (1.0 if neuro[1][i] in texto_lower else 0.0) for i in range(len(neuro[1]))]
    probs = softmax(ativacoes)
    idx_alvo = neuro[1].index(dogma_alvo) if dogma_alvo in neuro[1] else -1
    if idx_alvo >= 0:
        gradientes = [probs[i] - (1.0 if i == idx_alvo else 0.0) for i in range(len(probs))]
        for i in range(len(pesos)):
            pesos[i] = max(0.0001, pesos[i] - 0.05 * gradientes[i])
        total = sum(pesos)
        pesos = [p/total for p in pesos]

def ativa(texto):
    texto_lower = texto.lower()
    pos = 0
    dogmas_encontrados = []
    while pos < len(texto_lower):
        encontrou_algo = False
        for i, palavra in enumerate(neuro[1]):
            idx = texto_lower.find(palavra, pos)
            if idx == pos:
                id_acao = neuro[0][i]
                acao.append(id_acao)
                dogmas_encontrados.append(palavra)
                pos += len(palavra)
                encontrou_algo = True
                break
        if not encontrou_algo:
            pos += 1
    for dogma in dogmas_encontrados:
        ajustar_pesos(texto, dogma)
    return dogmas_encontrados

# ============================================================
# 2. TRADUTOR DETERMINÍSTICO
# ============================================================
class ManualTradutor:
    def __init__(self, filename="manual-python.txt"):
        self.manual_map = {}
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                # Captura sintaxes do manual
                try:
                    self.manual_map[1] = re.findall(r"#13\..*?\n(.*?\n)", content, re.S)[0].strip()
                    self.manual_map[2] = re.findall(r"#1\..*?\n(.*?\n)", content, re.S)[0].strip()
                    self.manual_map[3] = re.findall(r"#22\..*?\n(.*?\n)", content, re.S)[0].strip()
                except IndexError:
                    print("⚠️ Erro ao ler seções do manual. Verifique as tags #1, #13 e #22.")

    def limpar(self, linha, texto_original):
        palavras = texto_original.lower().split()
        nome_detectado = "Entidade"
        for i, p in enumerate(palavras):
            if p in ["class", "def"]:
                if i + 1 < len(palavras):
                    nome_detectado = palavras[i+1]
                    break
        
        linha = linha.replace("[nome]", nome_detectado).replace("[Nome]", nome_detectado.capitalize())
        if "valor" in texto_original.lower():
            partes = re.split(r"valor", texto_original, flags=re.IGNORECASE)
            if len(partes) > 1:
                conteudo = partes[1].strip()
                linha = linha.replace("[valor]", f"'{conteudo}'")
        
        linha = re.sub(r"\[.*?\]", "", linha)
        return linha.replace(", =", "").replace("() =", "()").replace("() :", "():").strip(), nome_detectado

# ============================================================
# 3. MONITOR VISUAL
# ============================================================
class Monitor:
    def __init__(self):
        self.root = None
        self.txt = None
        self.lbl_status = None

    def start(self):
        self.root = tk.Tk()
        self.root.title("QUINTIKUS PCODE - MONITOR")
        self.root.geometry("500x700")
        self.root.configure(bg="#050505")
        self.lbl_status = tk.Label(self.root, text="PONTEIRO: global", bg="#111", fg="#0f0", font=("Consolas", 10))
        self.lbl_status.pack(fill=tk.X)
        self.txt = scrolledtext.ScrolledText(self.root, bg="#050505", fg="#0f0", font=("Consolas", 12))
        self.txt.pack(fill=tk.BOTH, expand=True)
        self.root.mainloop()

    def log(self, msg, escopo):
        if self.root:
            self.txt.insert(tk.END, msg + "\n")
            self.lbl_status.config(text=f"PONTEIRO: {escopo}")
            self.txt.see(tk.END)

# ============================================================
# 4. GESTÃO DE ESTADO GLOBAL
# ============================================================
tradutor = ManualTradutor()
monitor = Monitor()
indent_level = 0
linha_atual = 0
escopo_stack = ["global"]
ultimo_nome = "global"
codigo_acumulado = []

def processar_pcode(entrada):
    global indent_level, acao, linha_atual, ultimo_nome
    acao.clear()
    ativa(entrada)
    
    for id_dogma in acao:
        if id_dogma == 4: # DENTRO
            indent_level += 1
            escopo_stack.append(ultimo_nome)
            continue
        
        if id_dogma == 5: # FORA
            if indent_level > 0:
                indent_level -= 1
                if len(escopo_stack) > 1:
                    escopo_stack.pop()
            continue
        
        sintaxe = tradutor.manual_map.get(id_dogma)
        if sintaxe:
            linha_pura, nome = tradutor.limpar(sintaxe, entrada)
            ultimo_nome = nome
            formatada = ("    " * indent_level) + linha_pura
            
            linha_atual += 1
            codigo_acumulado.append(formatada)
            monitor.log(formatada, escopo_stack[-1])

def main():
    global linha_atual, indent_level, escopo_stack, codigo_acumulado
    
    threading.Thread(target=monitor.start, daemon=True).start()
    print("🧠 QUINTIKUS v9.5 - PONTEIRO DE ESTADO ATIVO")
    
    while True:
        # Prompt dinâmico conforme sua lógica
        prompt = f"pcode [{escopo_stack[-1]} - linha:{linha_atual}] > "
        entrada = input(prompt).strip()
        
        if not entrada: continue
        if entrada.lower() == "exit": break
        
        if entrada.lower() == "run": 
            print("🚀 Executando...\n" + "-"*20)
            try: exec("\n".join(codigo_acumulado))
            except Exception as e: print(f"Erro: {e}")
            print("-"*20)
            continue
            
        if entrada.lower() == "clear":
             codigo_acumulado = []
             linha_atual = 0
             indent_level = 0
             escopo_stack = ["global"]
             print("🧹 Memória limpa.")
             continue
        
        processar_pcode(entrada)

if __name__ == "__main__":
    main()

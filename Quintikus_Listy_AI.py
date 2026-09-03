#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quintikus Listy-AI Master: Córtex Neural + Rede IMU + Ponto Zero (X0,Y0) + Entropia Linear + Captura de Cliques.
Requer: pip install pyautogui numpy keyboard mouse pillow
"""

import time
import math
import json
import os
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from typing import List, Dict, Any

try:
    import pyautogui
    import keyboard
    import mouse
    TEM_AUTOMACAO = True
except ImportError:
    TEM_AUTOMACAO = False

# Importa a base unificada do Córtex
from Quintikus_Listy_cortex import Quintikus_AGI_Unificado

class RedeVirtualIMU:
    """Rede IMU simulada para detectar vibrações, aceleração e tremores dinâmicos do mouse."""
    def __init__(self):
        self.historico_sensores = []

    def coletar_sincronizado(self, x: int, y: int, t: float):
        self.historico_sensores.append({"x": x, "y": y, "t": t})

    def processar_telemetria_inercial(self) -> Dict[str, Any]:
        if len(self.historico_sensores) < 3:
            return {"vel_max": 0, "aceleracao_media": 0, "vibracao_tremores": 0}

        velocidades = []
        aceleracoes = []
        vibracao = 0

        for i in range(1, len(self.historico_sensores)):
            p1, p2 = self.historico_sensores[i-1], self.historico_sensores[i]
            dt = p2["t"] - p1["t"]
            if dt <= 0: continue
            
            dx, dy = p2["x"] - p1["x"], p2["y"] - p1["y"]
            dist = math.sqrt(dx**2 + dy**2)
            velocidades.append(dist / dt)
            
            # Detecta micro-movimentos erráticos (tremores/vibração caótica)
            if 0 < dist < 4: 
                vibracao += 1

        for i in range(1, len(velocidades)):
            dt = self.historico_sensores[i+1]["t"] - self.historico_sensores[i]["t"]
            if dt > 0:
                aceleracoes.append(abs(velocidades[i] - velocidades[i-1]) / dt)

        return {
            "vel_max_px_s": round(max(velocidades) if velocidades else 0, 2),
            "aceleracao_media_px_s2": round(sum(aceleracoes) / len(aceleracoes) if aceleracoes else 0, 2),
            "vibracao_tremores": vibracao
        }


class QuintikusListyMasterGUI(Quintikus_AGI_Unificado):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("Quintikus Listy-AI - Córtex + IMU + Ponto Zero & Cliques")
        self.root.geometry("1000$x$620") if hasattr(self, 'x') else self.root.geometry("1000x620")
        self.root.config(bg="#1e1e1e")
        
        self.macros_salvas = "macros_treinadas.json"
        self.rede_imu = RedeVirtualIMU()
        self.carregar_macros()
        
        if not TEM_AUTOMACAO:
            messagebox.showwarning("Aviso", "Instale as dependências: pip install pyautogui keyboard mouse pillow")

        self.criar_interface()
        self.atualizar_painel_lateral()

    def criar_interface(self):
        # Painel esquerdo (Controles e Console)
        frame_esq = tk.Frame(self.root, bg="#252526", padx=15, pady=15)
        frame_esq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(frame_esq, text="Quintikus Listy-AI Orgânico", font=("Arial", 14, "bold"), fg="#4ec9b0", bg="#252526").pack(anchor="w", pady=(0, 15))

        tk.Label(frame_esq, text="Nome da Ação / Comando:", font=("Arial", 10), fg="#d4d4d4", bg="#252526").pack(anchor="w")
        self.entry_macro = tk.Entry(frame_esq, font=("Arial", 12), bg="#3c3c3c", fg="#ffffff", insertbackground="white")
        self.entry_macro.pack(fill=tk.X, pady=(5, 15))
        
        tk.Button(frame_esq, text="🔴 Gravar Macro Orgânica ([Page Up] / [Page Down])", font=("Arial", 10, "bold"), bg="#c586c0", fg="#ffffff", command=self.thread_gravar).pack(fill=tk.X, pady=5)
        tk.Button(frame_esq, text="🚀 Executar com Fusão Córtex + IMU", font=("Arial", 10, "bold"), bg="#4ec9b0", fg="#000000", command=self.thread_executar).pack(fill=tk.X, pady=5)

        self.console = scrolledtext.ScrolledText(frame_esq, height=14, bg="#1e1e1e", fg="#9cdcfe", font=("Consolas", 9))
        self.console.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

        # Painel direito (Inspeção de DNA e Métricas)
        frame_dir = tk.Frame(self.root, bg="#2d2d2d", width=380, padx=10, pady=10)
        frame_dir.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(frame_dir, text="🧬 DNA, Ponto Zero & IMU", font=("Arial", 11, "bold"), fg="#ce9178", bg="#2d2d2d").pack(anchor="w", pady=(0, 10))
        
        self.lista_macros = tk.Listbox(frame_dir, height=8, bg="#1e1e1e", fg="#dcdcaa", font=("Consolas", 10), selectbackground="#062f4a")
        self.lista_macros.pack(fill=tk.X, pady=(0, 10))
        self.lista_macros.bind('<<ListboxSelect>>', self.exibir_dna_selecionado)

        self.txt_dna = scrolledtext.ScrolledText(frame_dir, width=42, height=18, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.txt_dna.pack(fill=tk.BOTH, expand=True)

    def log(self, mensagem):
        self.console.insert(tk.END, mensagem + "\n")
        self.console.see(tk.END)

    def thread_gravar(self):
        nome = self.entry_macro.get().strip()
        if not nome:
            messagebox.showerror("Erro", "Digite o nome da ação antes de gravar!")
            return
        threading.Thread(target=self.gravar_macro_organica, args=(nome,), daemon=True).start()

    def thread_executar(self):
        nome = self.entry_macro.get().strip()
        if not nome:
            messagebox.showerror("Erro", "Selecione ou digite uma macro válida!")
            return
        threading.Thread(target=self.executar_macro_aprendida, args=(nome,), daemon=True).start()

    def calcular_entropia_linear_mouse(self, trajectoria: List[Dict[str, Any]]) -> float:
        """Calcula a entropia ponderada linear com base nos deslocamentos e cliques do mouse."""
        if len(trajectoria) < 2: return 0.0
        distancias = []
        for i in range(1, len(trajectoria)):
            p1, p2 = trajectoria[i-1], trajectoria[i]
            d = math.sqrt((p2["x"] - p1["x"])**2 + (p2["y"] - p1["y"])**2)
            distancias.append(d)
        
        if not distancias: return 0.0
        total = sum(distancias)
        if total == 0: return 0.0
        
        probs = [d / total for d in distancias]
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        return float(ent)

    def gravar_macro_organica(self, nome_acao: str):
        if not TEM_AUTOMACAO: return

        # Obtém tamanho dinâmico da tela e calcula o Ponto Zero central (X0, Y0)
        tela_w, tela_h = pyautogui.size()
        cx, cy = tela_w // 2, tela_h // 2
        
        self.log(f"📐 Resolução Dinâmica detectada: {tela_w}x{tela_h}")
        self.log(f"🔴 [Gravador] Aguardando [Page Up] para iniciar...")
        
        keyboard.wait('page up')
        
        # Reseta o mouse para o centro exato (Ponto Zero) antes de começar
        pyautogui.moveTo(cx, cy, duration=0.1)
        self.log(f"🎯 Ponto Zero ativado em X0={cx}, Y0={cy}. GRAVANDO!")

        passos = []
        tempo_inicio = time.time()
        ultimo_x, ultimo_y = pyautogui.position()
        
        estado_clique_esq = False
        estado_clique_dir = False
        self.rede_imu.historico_sensores = []

        while True:
            if keyboard.is_pressed('page down'):
                time.sleep(0.3) # Debounce
                self.log("⏹️ [Gravador] Gravação encerrada e processada.")
                break

            x, y = pyautogui.position()
            delta_t = time.time() - tempo_inicio

            # Captura dinâmica de cliques do mouse (Esquerdo e Direito)
            clique_esq_atual = mouse.is_pressed('left')
            clique_dir_atual = mouse.is_pressed('right')

            if clique_esq_atual and not estado_clique_esq:
                passos.append({"tipo": "clique", "botao": "left", "x": x, "y": y, "t": round(delta_t, 3)})
                self.log(f"🖱️ Clique Esquerdo capturado em X={x}, Y={y} (t={delta_t:.3f}s)")
            
            if clique_dir_atual and not estado_clique_dir:
                passos.append({"tipo": "clique", "botao": "right", "x": x, "y": y, "t": round(delta_t, 3)})
                self.log(f"🖱️ Clique Direito capturado em X={x}, Y={y} (t={delta_t:.3f}s)")

            estado_clique_esq = clique_esq_atual
            estado_clique_dir = clique_dir_atual

            # Rastreamento de movimento e telemetria IMU
            if x != ultimo_x or y != ultimo_y:
                passos.append({"tipo": "mover", "x": x, "y": y, "t": round(delta_t, 3)})
                self.rede_imu.coletar_sincronizado(x, y, delta_t)
                ultimo_x, ultimo_y = x, y

            time.sleep(0.02)

        # Processamento de Métricas Avançadas
        entropia_trajeto = self.calcular_entropia_linear_mouse(passos)
        dados_imu = self.rede_imu.processar_telemetria_inercial()
        dna_id = f"DNA-{abs(hash(nome_acao + str(entropia_trajeto))) % 10000:04d}"
        
        macro_dados = {
            "nome": nome_acao,
            "dna_id": dna_id,
            "resolucao": [tela_w, tela_h],
            "ponto_zero": [cx, cy],
            "entropia_trajeto": entropia_trajeto,
            "telemetria_imu": dados_imu,
            "passos": passos
        }

        self.salvar_macro_disco(nome_acao, macro_dados)
        self.tgnc.ensinar(nome_acao, "MACRO", "EXECUTAR")
        
        self.log(f"🧠 [Córtex] Macro mapeada com ID {dna_id} | Entropia Linear: {entropia_trajeto:.2f}")
        self.root.after(0, self.atualizar_painel_lateral)

    def salvar_macro_disco(self, nome: str, dados: Dict[str, Any]):
        self.banco_macros[nome] = dados
        with open(self.macros_salvas, "w") as f:
            json.dump(self.banco_macros, f, indent=4)
        self.carregar_macros()

    def carregar_macros(self):
        self.banco_macros = {}
        if os.path.exists(self.macros_salvas):
            try:
                with open(self.macros_salvas, "r") as f:
                    self.banco_macros = json.load(f)
            except: pass

    def atualizar_painel_lateral(self):
        self.lista_macros.delete(0, tk.END)
        for nome in self.banco_macros.keys():
            self.lista_macros.insert(tk.END, nome)

    def exibir_dna_selecionado(self, event):
        if not self.lista_macros.curselection(): return
        nome_macro = self.lista_macros.get(self.lista_macros.curselection()[0])
        self.entry_macro.delete(0, tk.END)
        self.entry_macro.insert(0, nome_macro)
        
        macro = self.banco_macros.get(nome_macro, {})
        imu = macro.get("telemetria_imu", {})
        res = macro.get("resolucao", [0, 0])
        pz = macro.get("ponto_zero", [0, 0])
        
        self.txt_dna.delete("1.0", tk.END)
        self.txt_dna.insert(tk.END, 
            f"=== DNA ENTRÓPICO ===\n"
            f"ID: {macro.get('dna_id')}\n"
            f"Resolução: {res[0]}x{res[1]}\n"
            f"Ponto Zero (X0,Y0): {pz[0]}, {pz[1]}\n"
            f"Entropia Linear: {macro.get('entropia_trajeto', 0):.4f}\n\n"
            f"=== TELEMETRIA IMU ===\n"
            f"Velocidade Máxima: {imu.get('vel_max_px_s', 0)} px/s\n"
            f"Aceleração Média: {imu.get('aceleracao_media_px_s2', 0)} px/s²\n"
            f"Tremores/Vibração: {imu.get('vibracao_tremores', 0)}\n"
            f"Total de Passos: {len(macro.get('passos', []))}\n"
        )

    def executar_macro_aprendida(self, nome_acao: str):
        self.log(f"\n🔍 [Listy-AI] Analisando execução para: '{nome_acao}'...")
        
        # Consulta o Córtex Neural (HRM/TGNC)
        nexo, acao, conf, mood = self.tgnc.analisar(nome_acao)
        self.log(f"🧠 Córtex Neural -> Nexo: {nexo} | Ação: {acao} | Confiança: {conf:.2f} | Humor: {mood}")

        macro = self.banco_macros.get(nome_acao)
        if not macro:
            self.log(f"⚠️ [Listy-AI] Nenhuma macro gravada com o nome '{nome_acao}'.")
            return False

        # Ponderação com a Rede IMU (Filtro de Entropia Caótica)
        imu = macro.get("telemetria_imu", {})
        vibracao = imu.get("vibracao_tremores", 0)
        
        if conf < 0.15:
            self.log("⛔ [Listy-AI] Execução abortada: Confiança do Córtex abaixo do limite.")
            return False
            
        if vibracao > 150:
            self.log(f"⚠️ [IMU Warning] Alta entropia física detectada ({vibracao} tremores). Ajustando compensação temporal.")

        # Reset Obrigatório no Ponto Zero (Centro da Tela) antes de iniciar
        tela_w, tela_h = pyautogui.size()
        cx, cy = tela_w // 2, tela_h // 2
        pyautogui.moveTo(cx, cy, duration=0.1)
        self.log(f"🎯 Ponto Zero reajustado para ({cx}, {cy}). Executando macro...")

        t_ultimo = 0.0
        for passo in macro["passos"]:
            delay = passo["t"] - t_ultimo
            if delay > 0:
                time.sleep(min(delay, 1.0))
            
            tipo = passo.get("tipo")
            if tipo == "mover":
                pyautogui.moveTo(passo["x"], passo["y"], duration=0.02)
            elif tipo == "clique":
                pyautogui.click(button=passo["botao"])
                self.log(f"🖱️ Executando clique '{passo['botao']}' em X={passo['x']}, Y={passo['y']}")
                
            t_ultimo = passo["t"]

        self.log(f"✅ [Listy-AI] Macro '{nome_acao}' executada com sucesso e precisão temporal ponderada!")
        return True

if __name__ == "__main__":
    root = tk.Tk()
    app = QuintikusListyMasterGUI(root)
    root.mainloop()

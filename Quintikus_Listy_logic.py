#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quintikus AGI Completo: Interface Tkinter + Visão (Entropia/Geometria/NCC) + Córtex Neural (TGNC v14)
Versão corrigida: NCC multi-escala (pirâmide) + Laminy para predição precisa.
"""

import base64
import ctypes
import json
import math
import os
import time
from io import BytesIO
from tkinter import (
    BOTH, HORIZONTAL, LEFT, RIGHT, VERTICAL, X, Y, NW,
    Canvas, Frame, Label, Entry, Scrollbar, Text, Tk, messagebox,
)
from PIL import Image, ImageTk
import keyboard
import pyautogui
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# === FIX === DPI do Windows (mantido)
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

os.environ["OMP_NUM_THREADS"] = "1"

# ==============================================================================
# 👁️ BLOCO 1: VISÃO — DNA ENTRÓPICO E GEOMETRIA TRIANGULAR
# ==============================================================================
VETOR_DIM = 32
RADA_N = 12


class DNAEntropico:
    def __init__(self, entropia, geometria, momentos, rada, linearidade,
                 entropia_dentro=0.0, entropia_fora=0.0):
        self.entropia = entropia
        self.geometria = geometria
        self.momentos = momentos
        self.rada = rada
        self.linearidade = linearidade
        self.entropia_dentro = entropia_dentro
        self.entropia_fora = entropia_fora
        self.hash = None
        self.size = None

    def to_array(self):
        return [self.entropia, self.geometria, self.momentos, self.rada,
                self.linearidade, self.entropia_dentro, self.entropia_fora]

    @staticmethod
    def from_array(a):
        return DNAEntropico(*a[:7])


class GeometriaTriangular:
    @staticmethod
    def pontos_entropia(pixels, largura, altura, n=3):
        bw = max(1, largura // 8)
        bh = max(1, altura // 8)
        pts = []
        for y in range(0, altura - bh + 1, bh):
            for x in range(0, largura - bw + 1, bw):
                bloco = []
                for j in range(bh):
                    for i in range(bw):
                        idx = (y + j) * largura + (x + i)
                        if idx < len(pixels):
                            bloco.append(pixels[idx])
                if not bloco:
                    continue
                hist = [0] * 256
                for p in bloco:
                    hist[p] += 1
                H = 0.0
                tot = len(bloco)
                for c in hist:
                    if c > 0:
                        p = c / tot
                        H -= p * math.log2(p)
                pts.append((H, x + (bw >> 1), y + (bh >> 1)))
        pts.sort(key=lambda p: p[0], reverse=True)
        return [[p[1], p[2]] for p in pts[:n]]

    @staticmethod
    def area(p1, p2, p3):
        return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0)

    @staticmethod
    def perimetro(p1, p2, p3):
        d = lambda a, b: math.hypot(a[0]-b[0], a[1]-b[1])
        return d(p1, p2) + d(p2, p3) + d(p3, p1)

    @staticmethod
    def angulos(p1, p2, p3):
        d = lambda a, b: math.hypot(a[0]-b[0], a[1]-b[1])
        a, b, cc = d(p2, p3), d(p1, p3), d(p1, p2)
        if not a or not b or not cc:
            return [0.0, 0.0, 0.0]
        A = math.acos(max(-1.0, min(1.0, (b*b + cc*cc - a*a) / (2*b*cc))))
        B = math.acos(max(-1.0, min(1.0, (a*a + cc*cc - b*b) / (2*a*cc))))
        return [A, B, math.pi - A - B]

    @staticmethod
    def momentos(pontos):
        if len(pontos) < 3:
            return [0.0, 0.0, 0.0, 0.0]
        p1, p2, p3 = pontos[:3]
        a = GeometriaTriangular.area(p1, p2, p3)
        p = GeometriaTriangular.perimetro(p1, p2, p3)
        g = GeometriaTriangular.angulos(p1, p2, p3)
        return [a / (1024 * 768), p / max(p, 1.0), g[0], g[1]]


class RADAMilitar:
    def __init__(self, n=RADA_N):
        self.n = n

    def calcular(self, pixels, largura, altura):
        acum = [0.0] * self.n
        step = max(1, min(largura, altura) // 20)
        for y in range(1, altura - 1, step):
            for x in range(1, largura - 1, step):
                i = y * largura + x
                gx = (pixels[min(i+1, len(pixels)-1)] - pixels[max(i-1, 0)]) / 2.0
                gy = (pixels[min(i+largura, len(pixels)-1)] - pixels[max(i-largura, 0)]) / 2.0
                if not gx and not gy:
                    continue
                ang = ((math.atan2(gy, gx) * 180 / math.pi) % 360 + 360) % 360
                mag = math.hypot(gx, gy)
                b = int((ang / (360 / self.n)) % self.n)
                acum[b] += mag
        mx = max(acum) if acum else 0.0
        e = [math.exp(v - mx) for v in acum]
        s = sum(e)
        return [v / s for v in e] if s > 0 else [1.0 / self.n] * self.n


class ConversorUniversal:
    @staticmethod
    def entropia(lst):
        hist = [0] * 256
        for p in lst:
            hist[p] += 1
        H = 0.0
        tot = len(lst)
        if tot == 0:
            return 0.0
        for c in hist:
            if c:
                p = c / tot
                H -= p * math.log2(p)
        return H

    def converter(self, pixels, largura, altura):
        redim = [0] * (VETOR_DIM * VETOR_DIM)
        for y in range(VETOR_DIM):
            for x in range(VETOR_DIM):
                sy = int(y * (altura / VETOR_DIM))
                sx = int(x * (largura / VETOR_DIM))
                redim[y * VETOR_DIM + x] = pixels[sy * largura + sx]

        dentro, fora = [], []
        for y in range(VETOR_DIM):
            for x in range(VETOR_DIM):
                v = redim[y * VETOR_DIM + x]
                if 8 <= x < 24 and 8 <= y < 24:
                    dentro.append(v)
                else:
                    fora.append(v)

        pts = GeometriaTriangular.pontos_entropia(pixels, largura, altura, 3)
        mean_val = sum(pixels) / len(pixels) / 255.0 if pixels else 0.0

        return {
            'entropia': self.entropia(pixels),
            'geometria': min(largura / altura, altura / largura) if altura > 0 and largura > 0 else 0,
            'momentos': [mean_val, 0.0, 0.0, 0.0],
            'momentos_tri': GeometriaTriangular.momentos(pts),
            'rada': RADAMilitar().calcular(pixels, largura, altura),
            'linearidade': 0.5,
            'entropia_dentro': self.entropia(dentro),
            'entropia_fora': self.entropia(fora),
            'size': [largura, altura],
        }


# ==============================================================================
# 🆕 BLOCO 1.5: MATCHING — NCC PIRÂMIDE + LAMINY (portado do JS)
# ==============================================================================

def laminy(d1, d2):
    """Similaridade Laminy entre dois DNAs (dicts). Retorna 0..1."""
    def col(a, b):
        return -math.log2(1 - max(0.0, min(1.0, abs(a - b))) + 1e-9)

    s = (3.0 * col(d1['entropia_dentro'] / 8, d2['entropia_dentro'] / 8) +
         3.0 * col(d1['entropia_fora'] / 8,   d2['entropia_fora'] / 8) +
              col(d1['entropia'] / 8,          d2['entropia'] / 8))

    n = min(len(d1['rada']), len(d2['rada']))
    for i in range(n):
        s += col(d1['rada'][i], d2['rada'][i]) / 12

    fator = 1.0 + abs(d1['geometria'] - d2['geometria'])
    return math.exp(-s / (10.0 * fator))


def ncc_map_small(screen_np, tmpl_np):
    """
    NCC map vetorizado via sliding_window_view.
    ⚠️ Use APENAS em arrays pequenos (ex: pirâmide fator 4). Em arrays grandes,
    a memória explode. Para fator 1, use `ncc_refine` local.
    """
    sh, sw = screen_np.shape
    th, tw = tmpl_np.shape
    if th >= sh or tw >= sw:
        return None
    tm = float(tmpl_np.mean())
    ts = float(tmpl_np.std())
    if ts < 1e-6:
        return None
    tz = (tmpl_np - tm).astype(np.float32)

    windows = sliding_window_view(screen_np, (th, tw)).astype(np.float32)
    means = windows.mean(axis=(2, 3))
    stds = np.sqrt(np.maximum(1e-6, windows.var(axis=(2, 3))))
    cross = np.tensordot(windows, tz, axes=([2, 3], [0, 1]))
    n = th * tw
    return cross / (n * stds * ts + 1e-9)


def ncc_at(screen_np, tz, ts, x, y):
    """NCC em uma posição específica (rápido, para refino local)."""
    th, tw = tz.shape
    sh, sw = screen_np.shape
    if x < 0 or y < 0 or x + tw > sw or y + th > sh:
        return -1.0
    window = screen_np[y:y+th, x:x+tw]
    wm = float(window.mean())
    ws = float(window.std())
    if ws < 1e-6:
        return 0.0
    n = th * tw
    return float(np.sum((window - wm) * tz) / (n * ws * ts + 1e-9))


def ncc_refine(screen_np, tmpl_np, cx, cy, radius=3):
    """Refina NCC localmente em torno de (cx, cy)."""
    tm = float(tmpl_np.mean())
    ts = float(tmpl_np.std())
    if ts < 1e-6:
        return cx, cy, 0.0
    tz = tmpl_np - tm
    best = (cx, cy, -1.0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            s = ncc_at(screen_np, tz, ts, cx + dx, cy + dy)
            if s > best[2]:
                best = (cx + dx, cy + dy, s)
    return best


def top_peaks(ncc_map, tmpl_shape, n=20, iou_thr=0.3):
    """Top-N posições (x, y, score) sem sobreposição."""
    th, tw = tmpl_shape
    flat = ncc_map.flatten()
    w = ncc_map.shape[1]
    order = np.argsort(flat)[::-1]
    peaks = []
    for idx in order:
        y, x = divmod(int(idx), w)
        ok = True
        for px, py, _ in peaks:
            x1 = max(x, px); y1 = max(y, py)
            x2 = min(x + tw, px + tw); y2 = min(y + th, py + th)
            if x2 <= x1 or y2 <= y1:
                continue
            inter = (x2 - x1) * (y2 - y1)
            iou = inter / (tw * th + tw * th - inter)
            if iou > iou_thr:
                ok = False
                break
        if ok:
            peaks.append((x, y, float(flat[idx])))
            if len(peaks) >= n:
                break
    return peaks


# ==============================================================================
# 🧠 BLOCO 2: CÓRTEX NEURAL (TGNC v14)
# ==============================================================================
def l2_normalize(x):
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8) if norm > 0 else x


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


class TGNC_NeuralCortex_v14:
    def __init__(self, dim=128, rank=64):
        self.dim, self.rank = dim, rank
        self.lr = 0.5
        self.vocab, self.concepts, self.actions = {}, {}, {}
        self.W1 = np.random.randn(dim, rank) * 0.01
        self.W2 = np.random.randn(rank, dim) * 0.01
        self.W_conf = np.random.randn(dim + dim, 1) * 0.01
        self.W_context, self.W_action = np.eye(dim), np.eye(dim)
        self.ruido = {"o", "a", "os", "as", "de", "da", "do", "um", "uma",
                      "com", "que", "para", "em", "no", "na", "e"}

    def _get_vector(self, word, space="vocab"):
        target = {"vocab": self.vocab, "concepts": self.concepts, "actions": self.actions}[space]
        if word not in target:
            target[word] = l2_normalize(np.random.uniform(-1, 1, self.dim))
        return target[word]

    def processar_sequencia(self, texto: str):
        if not texto or not texto.strip():
            return None, 0.0
        palavras = [w for w in texto.lower().replace(".", " ").replace(",", "").split()
                    if w not in self.ruido]
        if not palavras:
            mean_vec = np.mean([self._get_vector(p) for p in texto.lower().split()], axis=0)
            return l2_normalize(mean_vec), 0.15

        h_states, prev_v = [], np.zeros(self.dim)
        for p in palavras:
            v_k = self._get_vector(p)
            h_k = l2_normalize(v_k + (prev_v @ self.W1) @ self.W2)
            h_states.append(h_k)
            prev_v = h_k
        context_vector = l2_normalize(np.mean(h_states, axis=0))
        conf_score = float(sigmoid(np.dot(np.concatenate([context_vector, prev_v]),
                                          self.W_conf.flatten())))
        return context_vector, conf_score


# ==============================================================================
# 🖥️ BLOCO 3: INTERFACE GRÁFICA TKINTER E CONTROLADOR DO AGENTE
# ==============================================================================
class QuintikusAgenteAvancado:
    def __init__(self, root):
        self.root = root
        self.root.title("🧬 Quintikus — Agente Autônomo Unificado (Visão + Córtex)")
        self.root.geometry("1350x768")
        self.root.configure(bg="#0d1117")

        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.img_original = None
        self.ultimo_dna = None
        self.ultimo_recorte = None   # === NOVO === guarda o PIL Image do recorte (p/ NCC)
        self.banco_caminho = "banco_dna_quintikus.json"

        self.conversor = ConversorUniversal()
        self.tgnc = TGNC_NeuralCortex_v14()

        self.criar_interface()
        self.configurar_escuta_teclado()
        self.gerar_demo_inicial()

    def criar_interface(self):
        self.container_canvas = Frame(self.root, bg="#161b22")
        self.container_canvas.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

        self.canvas = Canvas(self.container_canvas, bg="#ffffff", cursor="cross")
        self.hbar = Scrollbar(self.container_canvas, orient=HORIZONTAL, command=self.canvas.xview)
        self.vbar = Scrollbar(self.container_canvas, orient=VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.hbar.pack(side="bottom", fill=X)
        self.vbar.pack(side=RIGHT, fill=Y)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        painel = Frame(self.root, bg="#161b22", width=380)
        painel.pack(side=RIGHT, fill=BOTH, expand=False, padx=10, pady=10)

        Label(painel, text="🧬 Quintikus Engine Pro", bg="#161b22", fg="#58a6ff",
              font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=10)

        Label(painel, text="Nome do Botão / Alvo:", bg="#161b22", fg="#8b949e",
              font=("Arial", 10)).pack(anchor="w", padx=10, pady=(5, 0))
        self.entry_nome = Entry(painel, bg="#0d1117", fg="#e6edf3",
                                insertbackground="white", font=("Arial", 11), relief="flat")
        self.entry_nome.pack(fill=X, padx=10, pady=5)
        self.entry_nome.insert(0, "botao_alvo_1")

        self.txt_info = Text(painel, bg="#0d1117", fg="#e6edf3", font=("Courier", 10),
                             height=16, width=42, bd=0)
        self.txt_info.pack(padx=10, pady=5)
        self.txt_info.insert("1.0",
            "1. Aperte [Print Screen] para capturar.\n"
            "2. Selecione o botão no canvas.\n"
            "3. Digite o nome e clique em Aprender.\n"
            "4. Para localizar, clique em Predict.")

        btn_aprender = Label(painel, text="🧠 1. Aprender DNA do Botão",
                             bg="#238636", fg="#ffffff", font=("Arial", 10, "bold"),
                             padx=10, pady=8, cursor="hand2")
        btn_aprender.pack(fill=X, padx=10, pady=5)
        btn_aprender.bind("<Button-1>", lambda e: self.aprender_botao())

        btn_predict = Label(painel, text="🎯 2. Predict Otimizado (Localizar)",
                            bg="#1f6feb", fg="#ffffff", font=("Arial", 10, "bold"),
                            padx=10, pady=8, cursor="hand2")
        btn_predict.pack(fill=X, padx=10, pady=5)
        btn_predict.bind("<Button-1>", lambda e: self.executar_predict())

    def configurar_escuta_teclado(self):
        try:
            keyboard.add_hotkey("print screen", self.capturar_tela_sistema)
        except Exception:
            pass

    def capturar_tela_sistema(self):
        self.root.iconify()
        self.root.update()
        time.sleep(0.2)
        screenshot = pyautogui.screenshot()
        self.root.deiconify()
        self.carregar_imagem(screenshot)

    def carregar_imagem(self, img_pil):
        self.img_original = img_pil
        img_w, img_h = img_pil.size
        self.canvas.config(scrollregion=(0, 0, img_w, img_h))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_img)
        self.atualizar_painel(f"📸 Tela capturada!\nResolução: {img_w}x{img_h}px")

    def gerar_demo_inicial(self):
        img = Image.new("RGB", (1366, 768), color="white")
        self.carregar_imagem(img)

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#58a6ff", width=2)

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        if not self.img_original:
            return
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        x1 = int(min(self.start_x, end_x))
        y1 = int(min(self.start_y, end_y))
        x2 = int(max(self.start_x, end_x))
        y2 = int(max(self.start_y, end_y))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.img_original.width, x2), min(self.img_original.height, y2)

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return

        recorte = self.img_original.crop((x1, y1, x2, y2))
        cinza = recorte.convert("L")
        pixels = list(cinza.getdata())
        w, h = recorte.size

        dados_conversao = self.conversor.converter(pixels, w, h)

        self.ultimo_dna = {
            "entropia": dados_conversao['entropia'],
            "geometria": dados_conversao['geometria'],
            "momentos": dados_conversao['momentos'] + dados_conversao['momentos_tri'],
            "rada": dados_conversao['rada'],
            "linearidade": dados_conversao['linearidade'],
            "entropia_dentro": dados_conversao['entropia_dentro'],
            "entropia_fora": dados_conversao['entropia_fora'],
            "size": [w, h],
            "coords_exemplo": [x1, y1, x2, y2],
        }
        self.ultimo_recorte = recorte.copy()   # === NOVO ===

        info = (
            f"🎯 Alvo Selecionado:\n"
            f"Box: [{x1}, {y1}, {x2}, {y2}]\n"
            f"Tam: {w}x{h}px\n"
            f"Entropia Global: {self.ultimo_dna['entropia']:.2f}\n"
            f"Entropia Dentro: {self.ultimo_dna['entropia_dentro']:.2f}\n"
            f"Entropia Fora:  {self.ultimo_dna['entropia_fora']:.2f}\n"
            f"-----------------------------------\n"
            f"Defina o nome ao lado e clique em Aprender!"
        )
        self.atualizar_painel(info)

    def aprender_botao(self):
        if not self.ultimo_dna or self.ultimo_recorte is None:
            messagebox.showwarning("Aviso", "Selecione um botão no canvas primeiro!")
            return

        # === FIX === antes usava pyautogui.time (não existe)
        nome_custom = self.entry_nome.get().strip()
        if not nome_custom:
            nome_custom = f"botao_alvo_{int(time.time())}"

        banco = []
        if os.path.exists(self.banco_caminho):
            try:
                with open(self.banco_caminho, "r", encoding="utf-8") as f:
                    banco = json.load(f).get("banco", [])
            except Exception:
                banco = []

        # === NOVO === serializa o recorte como PNG base64 para NCC futuro
        buffer = BytesIO()
        self.ultimo_recorte.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        item = {"label": nome_custom, "template_b64": img_b64, **self.ultimo_dna}
        banco.append(item)

        with open(self.banco_caminho, "w", encoding="utf-8") as f:
            json.dump({"banco": banco}, f, indent=4, ensure_ascii=False)

        self.atualizar_painel(
            f"🧠 Sucesso!\nSalvo com o nome: '{nome_custom}'\n"
            f"Banco: {self.banco_caminho}\nTotal: {len(banco)} alvos"
        )

    # ==========================================================================
    # === NOVO === PREDICT com NCC pirâmide + Laminy (fim do DNA-only)
    # ==========================================================================
    def executar_predict(self):
        if not os.path.exists(self.banco_caminho):
            messagebox.showwarning("Aviso", "Nenhum banco encontrado. Aprenda um botão primeiro.")
            return

        with open(self.banco_caminho, "r", encoding="utf-8") as f:
            banco = json.load(f).get("banco", [])

        if not banco:
            messagebox.showwarning("Aviso", "O banco está vazio.")
            return

        alvo = banco[-1]
        nome = alvo.get("label", "alvo")

        if "template_b64" not in alvo:
            messagebox.showerror("Erro",
                "Este alvo foi salvo numa versão antiga (sem template). "
                "Apague o banco e reaprenda.")
            return

        # Reconstrói o template do PNG base64
        tmpl_bytes = base64.b64decode(alvo["template_b64"])
        tmpl_img = Image.open(BytesIO(tmpl_bytes)).convert("L")
        tw, th = tmpl_img.size

        self.atualizar_painel(f"🔍 Buscando '{nome}' ({tw}×{th}px)...")
        self.root.update()

        t0 = time.time()

        # Tela em numpy (grayscale)
        tela = self.img_original.convert("L")
        sw, sh = tela.size
        tela_np = np.array(tela, dtype=np.float32)

        # ---- PIRÂMIDE: fator 4 ----
        F = 4
        tela_4 = np.array(tela.resize((max(4, sw // F), max(4, sh // F)), Image.LANCZOS),
                          dtype=np.float32)
        tmpl_4 = np.array(tmpl_img.resize((max(3, tw // F), max(3, th // F)), Image.LANCZOS),
                          dtype=np.float32)

        if tmpl_4.shape[0] >= tela_4.shape[0] or tmpl_4.shape[1] >= tela_4.shape[1]:
            messagebox.showwarning("Aviso",
                "Template muito grande para a pirâmide. Selecione uma área menor.")
            return

        mapa_4 = ncc_map_small(tela_4, tmpl_4)
        if mapa_4 is None:
            self.atualizar_painel("❌ Template uniforme (sem bordas). Reaprenda.")
            return

        peaks_4 = top_peaks(mapa_4, tmpl_4.shape, n=20, iou_thr=0.3)
        # Converte coords para tela cheia
        candidatos = [(px * F, py * F, sc) for px, py, sc in peaks_4]

        # ---- REFINO fator 2 ----
        tela_2 = np.array(tela.resize((max(4, sw // 2), max(4, sh // 2)), Image.LANCZOS),
                          dtype=np.float32)
        tmpl_2 = np.array(tmpl_img.resize((max(3, tw // 2), max(3, th // 2)), Image.LANCZOS),
                          dtype=np.float32)

        refino_2 = []
        for cx, cy, _ in candidatos:
            r = ncc_refine(tela_2, tmpl_2, cx // 2, cy // 2, radius=4)
            refino_2.append((r[0] * 2, r[1] * 2, r[2]))

        # ---- REFINO fator 1 (tela cheia) ----
        refino_1 = []
        for cx, cy, _ in refino_2:
            r = ncc_refine(tela_np, np.array(tmpl_img, dtype=np.float32),
                           cx, cy, radius=3)
            refino_1.append(r)

        refino_1.sort(key=lambda c: -c[2])
        if not refino_1:
            self.atualizar_painel("❌ Nenhum candidato encontrado.")
            return

        bx, by, score_ncc = refino_1[0]
        dt = (time.time() - t0) * 1000

        # ---- Confirmação por Laminy (DNA) ----
        if bx < 0 or by < 0 or bx + tw > sw or by + th > sh:
            self.atualizar_painel("❌ Candidato fora da tela.")
            return

        janela = self.img_original.crop((bx, by, bx + tw, by + th)).convert("L")
        px_janela = list(janela.getdata())
        dna_janela_conv = self.conversor.converter(px_janela, tw, th)

        alvo_dna = {
            "entropia": alvo["entropia"],
            "geometria": alvo["geometria"],
            "entropia_dentro": alvo["entropia_dentro"],
            "entropia_fora": alvo["entropia_fora"],
            "rada": alvo["rada"],
        }
        cand_dna = {
            "entropia": dna_janela_conv["entropia"],
            "geometria": dna_janela_conv["geometria"],
            "entropia_dentro": dna_janela_conv["entropia_dentro"],
            "entropia_fora": dna_janela_conv["entropia_fora"],
            "rada": dna_janela_conv["rada"],
        }
        score_laminy = laminy(alvo_dna, cand_dna)

        # ---- Desenha ----
        bx2, by2 = bx + tw, by + th
        self.canvas.create_rectangle(bx, by, bx2, by2, outline="#3fb950", width=3)
        self.canvas.create_text(
            bx, max(10, by - 14),
            text=f"🎯 {nome}  NCC={score_ncc:.0%}  DNA={score_laminy:.0%}",
            anchor="sw", fill="#3fb950", font=("Arial", 10, "bold"))

        # Diagnóstico do top-5
        top5 = "\n".join(
            f"  #{i+1}  X={int(c[0])} Y={int(c[1])}  NCC={c[2]:.1%}"
            for i, c in enumerate(refino_1[:5]))

        self.atualizar_painel(
            f"🎯 RESULTADO\n"
            f"Alvo:   {nome}\n"
            f"Posição: X={bx}  Y={by}\n"
            f"Tamanho: {tw}×{th}px\n"
            f"─────────────────────\n"
            f"NCC:    {score_ncc:.2%}\n"
            f"DNA:    {score_laminy:.2%}\n"
            f"Tempo:  {dt:.0f} ms\n"
            f"─────────────────────\n"
            f"Top-5 candidatos:\n{top5}"
        )

    def atualizar_painel(self, texto):
        self.txt_info.delete("1.0", "end")
        self.txt_info.insert("1.0", texto)


if __name__ == "__main__":
    root = Tk()
    app = QuintikusAgenteAvancado(root)
    root.mainloop()

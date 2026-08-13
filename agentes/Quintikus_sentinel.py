import psutil
import os
import time
import re
import platform
import json

# ====================================================
# 1. SISTEMA SENTINEL V4.0 - O GUARDIÃO FÍSICO
# ====================================================
class QuintikusSentinel:
    def __init__(self, cpu_limit=85.0, ram_limit=90.0, workspace=None):
        self.cpu_limit = cpu_limit
        self.ram_limit = ram_limit
        self.workspace = workspace or os.getcwd()
        self.start_time = time.time()
        self.action_history = []
        self.forbidden_patterns = [
            r"rm\s+-rf", r"format\s+", r"mkfs", r"os\.remove", 
            r"shutil\.rmtree", r"subprocess\.Popen\(.*shell=True",
            r"chmod\s+777", r"> /dev/sda", r"del\s+/f\s+/s"
        ]

    def capturar_vitals(self):
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        status = "ESTÁVEL"
        if cpu > self.cpu_limit or ram > self.ram_limit: status = "ALERTA"
        if cpu > 95.0: status = "CRÍTICO"
        return {"cpu_uso": f"{cpu}%", "ram_uso": f"{ram}%", "status": status}

    def validar_seguranca(self, tool_name, args):
        vitals = self.capturar_vitals()
        if vitals["status"] == "CRÍTICO":
            return False, f"🚨 KILL SWITCH: Hardware em risco ({vitals['cpu_uso']} CPU)."

        path_to_check = args.get('path') or args.get('file_path') or args.get('command')
        if path_to_check and isinstance(path_to_check, str):
            if ".." in path_to_check or (path_to_check.startswith("/") and not path_to_check.startswith(self.workspace)):
                 return False, f"❌ VIOLAÇÃO: Acesso negado fora do Workspace."

        str_args = str(args).lower()
        for pattern in self.forbidden_patterns:
            if re.search(pattern, str_args):
                return False, f"🚫 BLOQUEIO: Comando perigoso detectado."
        return True, "✅ Seguro"

    def registrar_log_evolutivo(self, tool, resultado):
        self.action_history.append({
            "t": tool, 
            "r": str(resultado)[:60] + "...", 
            "h": time.strftime("%H:%M:%S")
        })
        if len(self.action_history) > 5: self.action_history.pop(0)

    def gerar_relatorio_ia(self):
        v = self.capturar_vitals()
        h = self.action_history[-1] if self.action_history else "Nenhuma"
        return (f"\n[SENTINEL STATUS]\nHardware: CPU {v['cpu_uso']} | RAM {v['ram_uso']} | Status: {v['status']}\n"
                f"Última Ação: {h}\n----------------------------------------")

# ====================================================
# 2. SUBCONSCIENTE QUINTIKUS - A METACOGNIÇÃO
# ====================================================
class SubconscienteQuintikus:
    def __init__(self):
        self.erro_count = 0
        self.max_erros_antes_alerta = 5
        self.max_erros_antes_abortar = 10
        self.historico_erros = []

    def analisar_resultado(self, resultado):
        if any(indicator in str(resultado) for indicator in ["❌", "Erro", "falhou", "não encontrado", "Timeout"]):
            self.erro_count += 1
            self.historico_erros.append(str(resultado)[:100])
            if self.erro_count >= self.max_erros_antes_abortar: return "ABORTAR"
            if self.erro_count >= self.max_erros_antes_alerta: return "REPENSAR"
        else:
            self.erro_count = 0
            self.historico_erros = []
        return "CONTINUAR"

    def gerar_prompt_correcao(self):
        return (f"\n[🧠 ALERTA DO SUBCONSCIENTE]\nVocê falhou {self.erro_count} vezes. "
                f"PARE e mude sua estratégia! Verifique caminhos ou sintaxe.")

# ====================================================
# 3. FERRAMENTA TREE & LOOP INTEGRADO
# ====================================================
def list_dir_tree(path=".", level=0, prefix=""):
    try:
        output = ""
        ignore = [".git", "__pycache__", "venv", "node_modules"]
        if level == 0: output += f"📁 Raiz: {os.path.abspath(path)}\n"
        items = sorted(os.listdir(path))
        for i, item in enumerate(items):
            if item in ignore: continue
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            output += f"{prefix}{connector}{item}\n"
            if os.path.isdir(os.path.join(path, item)):
                output += list_dir_tree(os.path.join(path, item), level + 1, prefix + ("    " if is_last else "│   "))
        return output
    except Exception as e: return f"❌ Erro: {e}"

def run_loop_inteligente(task, sentinel, subconscious, ask_deepseek, execute_tool):
    print(f"🚀 Quintikus Core: Iniciando missão...")
    for i in range(20):
        vitals = sentinel.capturar_vitals()
        if vitals["status"] == "CRÍTICO": return "🚨 EMERGÊNCIA HARDWARE: Abortando."

        status_msg = ""
        if i > 0:
            last_res = sentinel.action_history[-1]["r"] if sentinel.action_history else ""
            decisao = subconscious.analisar_resultado(last_res)
            if decisao == "ABORTAR": return "❌ TAREFA CANCELADA: IA em loop de erro persistente."
            if decisao == "REPENSAR": status_msg = subconscious.gerar_prompt_correcao()

        prompt_ia = f"{sentinel.gerar_relatorio_ia()}\n{status_msg}\nMissão: {task}"
        response = ask_deepseek(prompt_ia)
        
        try:
            data = extract_json(response) # Presume-se a função existente
        except:
            subconscious.erro_count += 1
            continue

        if data.get("tool") is None: return data.get("final_answer")

        # VERIFICAÇÃO DE SEGURANÇA CROSS-CHECK
        seguro, motivo = sentinel.validar_seguranca(data["tool"], data["tool_args"])
        if not seguro:
            res = motivo
        else:
            if data["tool"] == "list_dir_tree": res = list_dir_tree()
            else: res = execute_tool(data["tool"], data["tool_args"])

        sentinel.registrar_log_evolutivo(data["tool"], res)
        print(f"Turno {i+1}: {data['tool']} -> {vitals['status']}")

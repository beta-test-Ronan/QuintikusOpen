# --- START OF FILE quintikus_sentinel.py ---
import psutil
import os
import platform

class QuintikusSentinel:
    def __init__(self):
        self.max_cpu_percent = 85.0  # Limite de segurança do i5
        self.max_ram_percent = 90.0
        self.workspace = os.getcwd()

    def get_system_status((self)):
        """Coleta a telemetria real do hardware."""
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        # Define o status de segurança
        safe_mode = "GREEN"
        if cpu > self.max_cpu_percent or ram > self.max_ram_percent:
            safe_mode = "YELLOW"
        if cpu > 95.0:
            safe_mode = "RED" # Bloqueio total

        return {
            "cpu_usage": cpu,
            "ram_usage": ram,
            "disk_free": 100 - disk,
            "os": platform.system(),
            "safety_level": safe_mode
        }

    def check_tool_safety(self, tool_name, args):
        """O 'Gatekeeper' que bloqueia ferramentas se o PC estiver em perigo."""
        status = self.get_system_status()
        
        # Bloqueio de Recursos
        if status["safety_level"] == "RED" and tool_name != "read_file":
            return False, "⚠️ BLOQUEIO DE EMERGÊNCIA: CPU em 95%. Operação abortada para proteger o hardware."

        # Bloqueio de Caminho (Path Traversal)
        if "file_path" in args or "path" in args:
            target = args.get("file_path") or args.get("path")
            if ".." in target or target.startswith("/") or ":" in target[1:]:
                # Impede que ele saia da pasta do projeto
                if not os.path.abspath(target).startswith(self.workspace):
                    return False, f"❌ VIOLAÇÃO DE SEGURANÇA: Tentativa de acessar fora do Workspace ({target})"

        return True, "✅ Seguro"

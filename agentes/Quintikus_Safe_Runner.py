import subprocess
import re
import time
import os
import ast

class QuintikusSafeRunner:
    def __init__(self):
        # "Tags" Inseguras: Padrões que o código não pode conter
        self.forbidden_tags = {
            "OS_COMMANDS": [r"os\.system", r"subprocess\.Popen", r"pty\.spawn"],
            "NETWORK_SENSITIVE": [r"socket\.", r"requests\.delete", r"urllib\."],
            "DANGEROUS_EVAL": [r"eval\(", r"exec\(", r"getattr"],
            "FILE_DELETION": [r"os\.remove", r"os\.rmdir", r"shutil\.rmtree"],
            "HARDCODED_SECRETS": [r"password =", r"api_key =", r"token ="]
        }

    def scan_code(self, code):
        """Verifica se o código contém tags inseguras antes da execução."""
        findings = []
        for tag, patterns in self.forbidden_tags.items():
            for pattern in patterns:
                if re.search(pattern, code):
                    findings.append(tag)
        return list(set(findings))

    def run_with_telemetry(self, code, filename="sandbox_test.py"):
        """Executa o código e retorna um relatório completo para a LLM."""
        
        # 1. Scanner de Segurança
        security_tags = self.scan_code(code)
        if security_tags:
            return {
                "status": "BLOCKED",
                "reason": f"Tags de segurança detectadas: {security_tags}",
                "telemetry": None
            }

        # 2. Preparação do arquivo
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        # 3. Execução Monitorada
        start_time = time.time()
        try:
            process = subprocess.run(
                ["python", filename],
                capture_output=True,
                text=True,
                timeout=5 # Timeout curto para segurança
            )
            execution_time = time.time() - start_time
            
            return {
                "status": "SUCCESS" if process.returncode == 0 else "FAILED",
                "stdout": process.stdout,
                "stderr": process.stderr,
                "telemetry": {
                    "duration_seconds": round(execution_time, 4),
                    "exit_code": process.returncode
                }
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "TIMEOUT",
                "reason": "O código excedeu o limite de 5 segundos.",
                "telemetry": None
            }
        finally:
            if os.path.exists(filename):
                os.remove(filename)

# Exemplo de uso:
#tester = QuintikusSafeRunner()
#report = tester.run_with_telemetry("""print('Olá Mundo')
#print('tudo ok!)""")
#print(report)

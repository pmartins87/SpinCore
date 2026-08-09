SpinCore R7.4 Ryzen Run Kit
===========================

1) Copie o CONTEUDO desta pasta para a raiz do repositorio SpinCore no Ryzen9.

   <SpinCore>\RUN_R7_4_RYZEN.bat
   <SpinCore>\RUN_R7_4_RYZEN.ps1
   <SpinCore>\tools\run_r7_4_ryzen.py

2) Execute RUN_R7_4_RYZEN.bat.

3) Resultado principal:
   validation\R7_4_RYZEN_REPORT.json

IMPORTANTE
---------
Este runner e fail-closed. Ele nao transforma benchmark sintetico em PASS do R7.4.
Para um PASS real, o checkout precisa conter fisicamente a pilha R6/R7 e o hook especifico do projeto:
  tools\r7_4_project_hook.py
ou
  tools\r7_4_calibrate_and_pilot.py

Se esses componentes nao existirem, o relatorio indicara FAIL_MISSING_R7_STACK
ou FAIL_MISSING_PROJECT_R7_4_HOOK em vez de fingir que a calibracao foi executada.

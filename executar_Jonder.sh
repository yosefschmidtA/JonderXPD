#!/bin/bash

# Este comando faz o script parar imediatamente se qualquer comando falhar.
set -e

# Imprime mensagens para sabermos em que etapa estamos
echo "--- (1/3) Executando: python3 diluir_arquivo_jonder.py ---"
python3 diluir_arquivo_jonder.py

echo ""
echo "--- (2/3) Executando: python3 read_lnls.py ---"
python3 read_lnls.py

echo ""
echo "--- (3/3) Executando: python3 leitorexp.py ---"
python3 leitorexp.py

echo ""
echo "--- Processo completo! Todos os scripts foram executados. ---"

import codecs
import json

arquivo = "data/historico_notas.json"  # ajuste o caminho se necessário

# Lê o arquivo byte a byte e remove o BOM
with open(arquivo, "rb") as f:
    conteudo = f.read()

# Remove qualquer marca BOM manualmente
conteudo = conteudo.replace(codecs.BOM_UTF8, b"").replace(codecs.BOM_UTF16_LE, b"").replace(codecs.BOM_UTF16_BE, b"")

# Salva novamente em UTF-8 puro
with open(arquivo, "wb") as f:
    f.write(conteudo)

# Testa se o JSON agora é válido
try:
    dados = json.loads(conteudo.decode("utf-8"))
    print("✅ JSON corrigido com sucesso!")
    print(dados)
except Exception as e:
    print("❌ Ainda há erro:", e)

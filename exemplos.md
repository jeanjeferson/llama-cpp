# Exemplo 1: Consulta básica com curl
curl -X POST http://localhost:8080/completion -H "Content-Type: application/json" -d '{
  "prompt": "Qual é a capital do Brasil?",
  "n_predict": 128,
  "temperature": 0.7,
  "stop": ["\n\n"]
}'

# Exemplo 2: Chat com Python
python3 -c '
import requests
import json

url = "http://localhost:8080/completion"
headers = {"Content-Type": "application/json"}
data = {
    "prompt": "<s>[INST] Quem foi Santos Dumont? [/INST]",
    "n_predict": 512,
    "temperature": 0.7,
    "stop": ["</s>"]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json()["content"])
'

# Exemplo 3: Consulta com Node.js
node -e '
const fetch = require("node-fetch");

async function askLlama() {
  const response = await fetch("http://localhost:8080/completion", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: "Explique o que é inteligência artificial em uma frase",
      n_predict: 64,
      temperature: 0.7
    })
  });
  
  const data = await response.json();
  console.log(data.content);
}

askLlama();
'




# Tornar o script de teste executável
chmod +x test_api.py

# Executar um teste básico
./test_api.py --prompt "Explique como funciona a inteligência artificial"

# Ajustar parâmetros
./test_api.py --prompt "Escreva um poema sobre tecnologia" --tokens 256 --temp 0.9
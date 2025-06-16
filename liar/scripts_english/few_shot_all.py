import json
import requests
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURATION ==========
ARQUIVOS_JSON = [
    Path("../dataset/test.jsonl"),
    Path("../dataset/train.jsonl"),
    Path("../dataset/valid.jsonl"),
]
CAMINHO_SAIDA = Path("../results_english/few_shot_all_en.json")
HOST = "http://localhost:11434/api/generate"
MODELO = "gemma3:4b"
# ===================================

EXAMPLES_FEW_SHOT = [
    # TRUE
    {
        "statement": "NASA landed a man on the Moon in 1969.",
        "subjects": "Space exploration",
        "speaker": "neil-armstrong",
        "speaker_job_title": "Astronaut",
        "state_info": "Ohio",
        "party_affiliation": "",
        "context": "Interview",
        "label": "true"
    },
    {
        "statement": "Hawaii became the 50th U.S. state in 1959.",
        "subjects": "History",
        "speaker": "historian",
        "speaker_job_title": "Professor",
        "state_info": "Hawaii",
        "party_affiliation": "",
        "context": "Lecture",
        "label": "true"
    },

    # FALSE
    {
        "statement": "Barack Obama was born in Kenya.",
        "subjects": "Politics",
        "speaker": "anonymous-conspiracist",
        "speaker_job_title": "Citizen",
        "state_info": "Texas",
        "party_affiliation": "Republican",
        "context": "Social media post",
        "label": "false"
    },
    {
        "statement": "COVID-19 was caused by 5G towers.",
        "subjects": "Science, Health",
        "speaker": "disinformer",
        "speaker_job_title": "Influencer",
        "state_info": "California",
        "party_affiliation": "Independent",
        "context": "Online video",
        "label": "false"
    },

    # HALF TRUE
    {
        "statement": "Canada offers free healthcare to everyone.",
        "subjects": "Health",
        "speaker": "political-analyst",
        "speaker_job_title": "Commentator",
        "state_info": "Ontario",
        "party_affiliation": "",
        "context": "TV show",
        "label": "half-true"
    },
    {
        "statement": "Renewable energy is the main source of power in Brazil.",
        "subjects": "Energy, Environment",
        "speaker": "environmental-engineer",
        "speaker_job_title": "Consultant",
        "state_info": "São Paulo",
        "party_affiliation": "",
        "context": "Interview",
        "label": "half-true"
    },

    # PANTS ON FIRE
    {
        "statement": "Vaccines implant tracking microchips.",
        "subjects": "Public health",
        "speaker": "conspiracy-theorist",
        "speaker_job_title": "Self-employed",
        "state_info": "Florida",
        "party_affiliation": "Republican",
        "context": "Viral video",
        "label": "pants-on-fire"
    },
    {
        "statement": "Drinking saltwater cures cancer.",
        "subjects": "Health",
        "speaker": "healer",
        "speaker_job_title": "Alternative therapist",
        "state_info": "Nevada",
        "party_affiliation": "",
        "context": "Alternative forum",
        "label": "pants-on-fire"
    },

    # MOSTLY FALSE
    {
        "statement": "Most refugees are criminals.",
        "subjects": "Immigration, Politics",
        "speaker": "politician-x",
        "speaker_job_title": "Congressman",
        "state_info": "Texas",
        "party_affiliation": "Republican",
        "context": "Rally",
        "label": "mostly-false"
    },
    {
        "statement": "Illegal immigrants have full access to Social Security.",
        "subjects": "Politics, Economy",
        "speaker": "debate-commentator",
        "speaker_job_title": "Analyst",
        "state_info": "Arizona",
        "party_affiliation": "",
        "context": "Televised debate",
        "label": "mostly-false"
    },

    # MOSTLY TRUE
    {
        "statement": "California has one of the strongest economies in the U.S.",
        "subjects": "Economy",
        "speaker": "governor",
        "speaker_job_title": "Governor",
        "state_info": "California",
        "party_affiliation": "Democrat",
        "context": "Press conference",
        "label": "mostly-true"
    },
    {
        "statement": "The U.S. spends more on healthcare per capita than any other country.",
        "subjects": "Health, Economy",
        "speaker": "economist",
        "speaker_job_title": "Researcher",
        "state_info": "New York",
        "party_affiliation": "",
        "context": "International conference",
        "label": "mostly-true"
    }
]

# Load existing results
if CAMINHO_SAIDA.exists():
    with open(CAMINHO_SAIDA, "r", encoding="utf-8") as f:
        saida_reduzida = json.load(f)
else:
    saida_reduzida = []

statements_processados = set(item["statement"] for item in saida_reduzida)

# Generate few-shot prompt in English
def gerar_prompt(dado):
    prompt = (
        "You are a fact-checker. Classify the claim based on the provided information.\n"
        "Use only the given information. Choose only one of the following options:\n"

        "true\n"
        "false\n"
        "half-true\n"
        "pants-fire\n"
        "barely-true\n"
        "mostly-true\n\n"

    )

    for ex in EXAMPLES_FEW_SHOT:
        prompt += f"Claim: \"{ex['statement']}\"\n"
        prompt += f"Subject(s): {ex['subjects']}\n"
        prompt += f"Who made the claim: {ex['speaker']} ({ex['speaker_job_title']}, {ex['party_affiliation']} - {ex['state_info']})\n"
        prompt += f"Context: {ex['context']}\n"
        prompt += f"Label: {ex['label']}\n\n"

    prompt += f"Claim: \"{dado['statement']}\"\n"
    prompt += f"Subject(s): {dado.get('subjects', '')}\n"
    prompt += f"Who made the claim: {dado.get('speaker', '')} ({dado.get('speaker_job_title', '')}, {dado.get('party_affiliation', '')} - {dado.get('state_info', '')})\n"
    prompt += f"Context: {dado.get('context', '')}\n"
    prompt += "Label:"
    return prompt

# Send prompt to model
def classificar_claim(modelo, prompt):
    try:
        response = requests.post(
            HOST,
            json={"model": modelo, "prompt": prompt},
            timeout=90,
            stream=True
        )
        partes = []
        for linha in response.iter_lines(decode_unicode=True):
            if not linha.strip():
                continue
            try:
                obj = json.loads(linha)
                if "response" in obj:
                    partes.append(obj["response"])
                if obj.get("done", False):
                    break
            except json.JSONDecodeError:
                continue

        resposta_final = "".join(partes).strip()
        return resposta_final if resposta_final else "Empty response"

    except Exception as e:
        print(f"Error during classification: {e}")
        return "Connection error"

# Process each file
for arquivo in ARQUIVOS_JSON:
    with open(arquivo, "r", encoding="utf-8") as f:
        for linha in tqdm(f, desc=f"Processing {arquivo.name}"):
            dado = json.loads(linha)
            statement = dado["statement"]

            if statement in statements_processados:
                continue

            prompt = gerar_prompt(dado)
            label = classificar_claim(MODELO, prompt)

            saida_reduzida.append({
                "statement": statement,
                "label": label.strip().lower().strip('"')
            })
            statements_processados.add(statement)

# Save output
with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f:
    json.dump(saida_reduzida, f, indent=2, ensure_ascii=False)

print(f"\n💾 Results saved to: {CAMINHO_SAIDA}")
